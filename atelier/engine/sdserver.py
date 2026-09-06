"""Moteur RÉSIDENT : le même sd.cpp, mais qui ne meurt pas entre deux images.

`sd-cli` est un programme en ligne de commande : il démarre, charge le modèle,
produit l'image et meurt. Sur un journal réel, ça faisait 80 s de lecture de
modèle et 38 s d'encodage repayées à CHAQUE image — 118 s de frais fixes pour
117 s de calcul utile. Quand on génère une image à la fois pour tâtonner sur un
prompt, c'est la moitié du temps qui part là.

`sd-server` garde le modèle chargé et répond à des requêtes HTTP locales.
Une nouvelle image évite le rechargement des poids ; un nouveau prompt doit
encore être encodé. Aucun gain GPU chiffré n'est garanti par cette intégration.

Ce que ça coûte, et qu'il faut assumer :
  • le modèle occupe la VRAM en permanence -> tout outil qui a besoin du GPU
    doit d'abord appeler `stop()` (c'est fait dans `tools._run_tool`) ;
  • l'API n'expose pas d'aperçu en cours de génération : l'image apparaît d'un
    coup, à la fin ;
  • un modèle par serveur : changer de modèle relance le processus.

D'où le principe de cette intégration : le serveur n'est JAMAIS obligatoire.
Il est optionnel, il ne traite que les cas qu'il sait traiter, et la moindre
anomalie renvoie l'appelant vers `sd-cli`, qui reste la référence.
"""
from __future__ import annotations

import atexit
import base64
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import settings
from . import sdcpp

SERVER_NAMES = ("sd-server.exe", "sd-server")
# Le chargement, lui, n'a pas accéléré : sur un disque mécanique un modèle de
# 8 Go met plus d'une minute. Ce délai couvre le pire cas observé, largement.
READY_TIMEOUT_S = 900.0
_POLL_S = 0.4
_HTTP_TIMEOUT_S = 30.0
JOB_TIMEOUT_S = 7200.0
# This client only talks to 127.0.0.1. A system proxy must not receive images.
_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ServerUnavailable(RuntimeError):
    """Le moteur résident ne peut pas servir cette demande. Repli sur sd-cli."""


def find_server() -> Path | None:
    """Binaire sd-server, livré dans la même archive que sd-cli."""
    for name in SERVER_NAMES:
        for candidate in settings.BIN_DIR.rglob(name):
            if candidate.is_file():
                return candidate
    return None


def available() -> bool:
    return find_server() is not None


def can_serve(req: "sdcpp.GenRequest") -> bool:
    """Cette demande est-elle dans le périmètre sûr du serveur ?

    On refuse tout ce dont le comportement n'est pas vérifié à l'identique :
    les LoRA (l'API ignore délibérément les balises `<lora:…>` du prompt), la
    passe HD, les caches inter-pas et l'auto-fit. Les références sont acceptées
    seulement si l'API annonce cette capacité au moment de la demande.
    Ce n'est pas une limite définitive, c'est la liste de ce qui n'a pas encore
    été mesuré — et un repli silencieux vaut mieux qu'une image fausse.
    """
    if req.lora_dir or req.hires or req.auto_fit:
        return False
    if req.cache_mode:
        return False
    return bool(req.diffusion_model or req.model_path)


# --------------------------------------------------------------------------- #
#  Processus
# --------------------------------------------------------------------------- #
@dataclass
class _Live:
    proc: subprocess.Popen
    port: int
    key: str


_LIVE: _Live | None = None
_LOCK = threading.RLock()
_TAIL: deque[str] = deque(maxlen=200)
_SINK: Callable[[str], None] | None = None
_JOB: str | None = None
_GENERATION_LOCK = threading.Lock()
_CANCEL = threading.Event()


def _file_signature(values: list[str]) -> list[tuple]:
    """Replacing weights or the binary at the same path invalidates residency."""
    signatures = []
    for value in values:
        try:
            path = Path(value)
            if path.is_file():
                stat = path.stat()
                signatures.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        except OSError:
            pass
    return signatures


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pump(proc: subprocess.Popen) -> None:
    """Draine la sortie du serveur en continu.

    Un tube jamais lu finit par se remplir et bloquer le processus qui écrit
    dedans : ce fil n'est pas un confort de journalisation, c'est ce qui
    empêche le serveur de se figer au bout de quelques milliers de lignes.
    """
    disk = sdcpp.DiskWatch()
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip("\n")
        _TAIL.append(text)
        sink = _SINK
        if sink:
            try:
                sink(text)
                note = disk.note(text)
                if note:
                    sink(note)
            except Exception:  # noqa: BLE001
                pass


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/sdcpp/v1"


def _request(url: str, payload: dict | None = None,
             timeout: float = _HTTP_TIMEOUT_S) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with _HTTP.open(req, timeout=timeout) as response:
        body = response.read()
    parsed = json.loads(body.decode("utf-8")) if body else {}
    if not isinstance(parsed, dict):
        raise ServerUnavailable("The engine returned an invalid response.")
    return parsed


def is_running() -> bool:
    with _LOCK:
        return _LIVE is not None and _LIVE.proc.poll() is None


def stop(reason: str = "", log: Callable[[str], None] | None = None) -> None:
    """Arrête le serveur et rend la VRAM. Sans effet s'il ne tourne pas."""
    global _LIVE, _SINK
    with _LOCK:
        live, _LIVE, _SINK = _LIVE, None, None
    if live is None:
        return
    if log:
        log(f"⏹️ Resident engine stopped{(' : ' + reason) if reason else ''}.")
    try:
        live.proc.terminate()
        live.proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            live.proc.kill()
            live.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


atexit.register(stop)


def _wait_ready(port: int, proc: subprocess.Popen,
                log: Callable[[str], None] | None) -> None:
    url = _base_url(port) + "/capabilities"
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if _CANCEL.is_set():
            raise sdcpp.EngineError("Interrupted by the user.")
        if proc.poll() is not None:
            raise ServerUnavailable(
                "The resident engine stopped while starting up: "
                + " | ".join(list(_TAIL)[-3:]))
        try:
            _request(url, timeout=3.0)
            return
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(_POLL_S)
    raise ServerUnavailable("The resident engine did not answer in time.")


def ensure(server: Path, args: list[str], gpu_index: int | None,
           all_gpus: bool, log: Callable[[str], None] | None = None) -> int:
    """Garantit qu'un serveur tourne AVEC CES paramètres-là. Renvoie son port.

    La clé est la ligne de commande complète : changer de modèle, de quant, de
    résidence ou de carte relance le processus. C'est voulu — un serveur qui
    servirait un autre modèle que celui demandé serait bien pire que lent.
    """
    global _LIVE, _SINK
    key = json.dumps([str(server), args, gpu_index, all_gpus,
                      _file_signature([str(server), *args])])
    with _LOCK:
        if _LIVE is not None and _LIVE.proc.poll() is None and _LIVE.key == key:
            _SINK = log
            return _LIVE.port
        if _LIVE is not None:
            stop("model or settings changed", log)

        port = _free_port()
        cmd = [str(server), *args, "--listen-ip", "127.0.0.1",
               "--listen-port", str(port), "-v"]
        env = sdcpp.child_env_for(gpu_index, all_gpus)
        if log:
            log("$ " + " ".join(cmd))
            log("⏳ First start: the model loads once and for all. The images "
                "that follow will not wait.")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, cwd=str(settings.ROOT), env=env,
            encoding="utf-8", errors="replace")
        _TAIL.clear()
        _SINK = log
        threading.Thread(target=_pump, args=(proc,), daemon=True).start()
        _LIVE = _Live(proc=proc, port=port, key=key)
    try:
        _wait_ready(port, proc, log)
    except ServerUnavailable:
        stop()
        raise
    return port


# --------------------------------------------------------------------------- #
#  Génération
# --------------------------------------------------------------------------- #
def server_args(server: Path, req: "sdcpp.GenRequest") -> list[str]:
    """Arguments de DÉMARRAGE : ce qui décrit le modèle et la mémoire.

    Tout ce qui change d'une image à l'autre (prompt, taille, graine…) part
    dans la requête HTTP, pas ici — c'est cette séparation qui permet au modèle
    de rester chargé.
    """
    args: list[str] = []
    if req.model_path:
        args += ["-m", str(req.model_path)]
        if req.vae:
            args += ["--vae", str(req.vae)]
    else:
        args += ["--diffusion-model", str(req.diffusion_model)]
        if req.uncond_model:
            args += ["--uncond-diffusion-model", str(req.uncond_model)]
        if req.vae:
            args += ["--vae", str(req.vae)]
        if req.text_encoder:
            args += ["--llm", str(req.text_encoder)]
        if req.llm_vision:
            args += ["--llm_vision", str(req.llm_vision)]
        if req.t5xxl:
            args += ["--t5xxl", str(req.t5xxl)]
        if req.clip_l:
            args += ["--clip_l", str(req.clip_l)]
    args += list(req.extra_flags)

    args += sdcpp.memory_args(server, req)
    return args


def _b64(path: Path | None) -> str | None:
    if not path:
        return None
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def request_payload(req: "sdcpp.GenRequest") -> dict:
    """La demande d'image, telle que l'API native l'attend."""
    sample: dict = {"sample_method": req.sampler, "sample_steps": int(req.steps),
                    "guidance": {"txt_cfg": float(req.cfg_scale)}}
    if req.schedule:
        sample["scheduler"] = req.schedule
    if req.flow_shift and req.flow_shift > 0:
        sample["flow_shift"] = float(req.flow_shift)
    payload: dict = {
        "prompt": req.prompt,
        "width": int(req.width), "height": int(req.height),
        "seed": int(req.seed), "batch_count": int(req.batch_count),
        "sample_params": sample, "output_format": "png",
    }
    # Le négatif n'a de sens qu'avec un CFG > 1 : en dessous, sd.cpp ne calcule
    # pas la branche non conditionnée. Même règle que la ligne de commande.
    if req.negative and req.cfg_scale > 1.0:
        payload["negative_prompt"] = req.negative
    if req.init_image:
        payload["init_image"] = _b64(req.init_image)
        payload["strength"] = float(req.strength)
        if req.mask_image:
            payload["mask_image"] = _b64(req.mask_image)
    if req.ref_image:
        payload["ref_images"] = [_b64(p) for p in sdcpp._ref_list(req.ref_image)]
    return payload


def _write_images(images: list[dict], output: Path,
                  batch_count: int) -> list[Path]:
    """Écrit les images comme le ferait sd-cli, pour que la suite ne voie rien.

    `collect_outputs` cherche « <stem>_*.png » au-delà d'une image : on garde
    exactement cette convention plutôt que d'inventer la nôtre.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, item in enumerate(images):
        raw = item.get("b64_json") or ""
        if not raw:
            continue
        dest = (output if batch_count <= 1 and len(images) == 1
                else output.with_name(f"{output.stem}_{i + 1}{output.suffix}"))
        dest.write_bytes(base64.b64decode(raw))
        written.append(dest)
    return written


def cancel_active() -> str:
    """Native API cannot interrupt generating jobs; terminate to release the GPU."""
    with _LOCK:
        live = _LIVE
    if live is None:
        return ""
    _CANCEL.set()
    stop()
    return "⏹️ Generation cancelled."


def generate(server: Path, req: "sdcpp.GenRequest", output: Path,
             gpu_index: int | None = None, all_gpus: bool = False,
             log: Callable[[str], None] | None = None) -> list[Path]:
    """One job owns residency; cancellation must never trigger a CLI retry."""
    if not _GENERATION_LOCK.acquire(blocking=False):
        raise sdcpp.EngineError("The resident engine is already generating an image.")
    _CANCEL.clear()
    try:
        return _generate(server, req, output, gpu_index, all_gpus, log)
    except (ServerUnavailable, OSError, ValueError) as exc:
        if _CANCEL.is_set():
            raise sdcpp.EngineError("Interrupted by the user.") from exc
        if isinstance(exc, ServerUnavailable):
            raise
        raise ServerUnavailable(f"Resident engine unavailable: {exc}") from exc
    finally:
        _GENERATION_LOCK.release()


def _generate(server: Path, req: "sdcpp.GenRequest", output: Path,
              gpu_index: int | None, all_gpus: bool,
              log: Callable[[str], None] | None) -> list[Path]:
    global _JOB
    if not can_serve(req):
        raise ServerUnavailable("Request outside the resident engine's scope.")
    port = ensure(server, server_args(server, req), gpu_index, all_gpus, log)
    base = _base_url(port)
    if req.ref_image:
        caps = _request(f"{base}/capabilities")
        features = (caps.get("features_by_mode") or {}).get("img_gen") or caps.get("features") or {}
        if not features.get("ref_images"):
            raise ServerUnavailable("This server does not support reference images.")
    if _CANCEL.is_set():
        raise sdcpp.EngineError("Interrupted by the user.")
    try:
        job = _request(f"{base}/img_gen", payload=request_payload(req),
                       timeout=60.0)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise ServerUnavailable(f"Request refused by the server: {exc}")
    job_id = job.get("id")
    if not job_id:
        raise ServerUnavailable("The server opened no job.")
    with _LOCK:
        _JOB = job_id

    try:
        deadline = time.monotonic() + JOB_TIMEOUT_S
        while True:
            if _CANCEL.is_set():
                raise sdcpp.EngineError("Interrupted by the user.")
            if time.monotonic() >= deadline:
                stop()
                raise sdcpp.EngineError("Resident generation timed out; the engine was stopped.")
            time.sleep(_POLL_S)
            try:
                state = _request(f"{base}/jobs/{job_id}")
            except (urllib.error.URLError, OSError) as exc:
                raise ServerUnavailable(f"Lost track of the job: {exc}")
            status = state.get("status") or ""
            if status not in {"queued", "generating", "completed", "cancelled", "failed"}:
                raise ServerUnavailable(f"Unknown job status: {status!r}")
            if status == "completed":
                images = ((state.get("result") or {}).get("images")) or []
                written = _write_images(images, output, req.batch_count)
                if not written:
                    raise ServerUnavailable("The job finished with no image.")
                return written
            if status == "cancelled":
                raise sdcpp.EngineError("Interrupted by the user.")
            if status == "failed":
                error = (state.get("error") or {}).get("message") or "inconnue"
                if any(marker.lower() in error.lower() for marker in sdcpp._OOM_MARKERS):
                    raise sdcpp.VramError(f"The engine ran out of memory: {error}")
                # Un échec de GÉNÉRATION n'est pas un échec du serveur : le
                # relancer en ligne de commande donnerait la même erreur, et
                # avec 80 s de chargement en plus.
                raise sdcpp.EngineError(f"The engine failed: {error}")
    finally:
        with _LOCK:
            _JOB = None
