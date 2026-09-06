#!/usr/bin/env python3
"""Runner de restauration de visages (CodeFormer).

Ni `basicsr` ni le dépôt CodeFormer d'origine : basicsr importe
``torchvision.transforms.functional_tensor``, supprimé de torchvision depuis la
0.17 — il ne s'installe plus sur notre socle (torch 2.4.1 / torchvision 0.19) et
le corriger reviendrait à maintenir un patch à vie. À la place :

  • l'architecture vient de **spandrel** (pur torch, maintenu, sans registre
    global ni fichier de config à trimballer) ;
  • la détection, l'alignement sur le gabarit FFHQ 512 et le recollage avec
    masque de segmentation viennent de **facexlib**, qui est exactement ce
    qu'utilise le CodeFormer officiel pour cette partie.

Le résultat est donc le pipeline de référence, avec deux dépendances vivantes au
lieu d'une morte. Lancé en sous-process pour ne pas verrouiller les DLL de torch
dans Gradio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import label, pick_device  # noqa: E402

# Sortie du réseau : RGB dans [-1, 1]. On repasse en BGR uint8 parce que tout le
# reste du pipeline facexlib (recollage compris) travaille en BGR, comme OpenCV.
def _unit_to_bgr_uint8(tensor):
    """Sortie spandrel : RGB dans [0, 1] -> BGR uint8 (convention OpenCV)."""
    out = tensor.squeeze(0).detach().float().cpu().clamp_(0, 1)
    arr = out.numpy().transpose(1, 2, 0)[:, :, ::-1]
    return (arr * 255.0).round().astype("uint8")


def _to_bgr_uint8(tensor):
    out = tensor.squeeze(0).detach().float().cpu().clamp_(-1, 1)
    out = (out + 1.0) / 2.0
    arr = out.numpy().transpose(1, 2, 0)[:, :, ::-1]
    return (arr * 255.0).round().astype("uint8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--weights", default="GFPGANv1.4.pth",
                    help="fichier du restaurateur, dans --model-dir")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    # 0 = le modèle recrée librement (visages très abîmés), 1 = il colle au
    # pixel d'origine. La valeur du CodeFormer officiel est 0.5.
    ap.add_argument("--fidelity", type=float, default=0.5)
    ap.add_argument("--only-center", action="store_true",
                    help="ne restaurer que le visage le plus central")
    args = ap.parse_args()

    import cv2
    import torch
    from torchvision.transforms.functional import normalize
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from facexlib.utils.misc import img2tensor
        import spandrel
        import spandrel_extra_arches
    except ImportError as exc:
        sys.exit(f"Dépendance manquante ({exc}). Réinstallez l'outil "
                 "« Visages » depuis le Toolkit.")

    model_dir = Path(args.model_dir)
    # `name` : on refuse tout chemin — un « ../.. » dans l'argument sortirait
    # du dossier des modèles.
    weights = model_dir / Path(args.weights).name
    if not weights.is_file():
        sys.exit(f"Poids introuvables : {weights}")

    dev = pick_device(torch)
    device = torch.device(dev)
    print(f"[face] chargement de CodeFormer sur {label(dev)}…", flush=True)

    # CodeFormer est sous licence non commerciale : spandrel le range donc dans
    # le paquet « extra_arches », qu'il faut enregistrer explicitement. GFPGAN
    # et RestoreFormer, eux, sont dans le paquet principal.
    spandrel_extra_arches.install(ignore_duplicates=True)
    descriptor = spandrel.ModelLoader().load_from_file(str(weights))
    arch = descriptor.architecture.id
    if arch not in ("CodeFormer", "GFPGAN", "RestoreFormer"):
        sys.exit(f"Ce fichier n'est pas un restaurateur de visages reconnu "
                 f"({arch}).")
    descriptor.to(device).eval()
    net = descriptor.model
    print(f"[face] architecture : {arch}", flush=True)

    helper = FaceRestoreHelper(
        1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png",
        use_parse=True, device=device, model_rootpath=str(model_dir))

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Image illisible : {args.input}")
    helper.read_image(img)
    # `resize=640` : la détection tourne sur une version réduite (rapide et
    # suffisante), les points sont ensuite remis à l'échelle de l'original.
    count = helper.get_face_landmarks_5(
        only_center_face=bool(args.only_center), resize=640,
        eye_dist_threshold=5)
    if not count:
        sys.exit("Aucun visage détecté sur cette image.")
    print(f"[face] {count} visage(s) détecté(s).", flush=True)
    helper.align_warp_face()

    fidelity = min(1.0, max(0.0, float(args.fidelity)))
    if arch != "CodeFormer" and abs(fidelity - 0.5) > 0.01:
        print("[face] ce modèle n'a pas de curseur de fidélité : ignoré.",
              flush=True)
    for index, cropped in enumerate(helper.cropped_faces, 1):
        face = img2tensor(cropped / 255.0, bgr2rgb=True,
                          float32=True).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                if arch == "CodeFormer":
                    # Le pipeline officiel travaille en [-1, 1] et c'est le
                    # SEUL des trois à exposer le « w » de fidélité. spandrel
                    # ne le passe pas : on appelle le réseau directement.
                    normalize(face[0], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                              inplace=True)
                    restored = _to_bgr_uint8(net(face, weight=fidelity)[0])
                else:
                    # spandrel connaît la convention de CHAQUE architecture
                    # (RestoreFormer normalise en interne, GFPGAN non) : on lui
                    # donne du [0, 1] et il rend du [0, 1].
                    restored = _unit_to_bgr_uint8(descriptor(face))
        except RuntimeError as exc:
            # Un visage raté ne doit pas perdre les autres : on recolle
            # l'original à sa place et on le dit.
            print(f"[face] visage {index} non restauré ({exc}).", flush=True)
            restored = cropped.astype("uint8")
        helper.add_restored_face(restored)
        print(f"[face] visage {index}/{count} restauré.", flush=True)

    helper.get_inverse_affine(None)
    # `use_parse=True` : le masque suit la segmentation du visage (peau, yeux,
    # bouche) au lieu d'un simple rectangle — c'est ce qui évite la vignette
    # rectangulaire visible autour des visages recollés.
    result = helper.paste_faces_to_input_image()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_face.png")
    cv2.imwrite(str(dest), result)
    print(f"[face] image écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
