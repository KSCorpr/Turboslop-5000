"""Écriture de fichiers PSD à calques — Python pur, aucune dépendance compilée.

Pourquoi maison plutôt qu'une bibliothèque : il n'en existe pas d'utilisable.
`psd-tools` lit très bien mais ne sait PAS ajouter de calque (« does not support
editing of layer structure »), et `pytoshop` — le seul qui écrivait — date de
2018 et ne se compile plus sur un Python moderne. Or le format est documenté et
n'a besoin de rien d'autre que `struct` : l'écrire nous-mêmes coûte moins cher
qu'une dépendance morte, et ça tourne tel quel dans le Python portable.

Deux décisions qui font toute la différence sur la taille du fichier :

  • chaque calque est RECADRÉ sur sa boîte englobante. Un PSD stocke la position
    du calque, pas une image pleine toile : garder la toile entière pour un
    objet de 200 px multiplierait le poids par vingt ;
  • les canaux sont compressés en RLE PackBits, l'encodage natif du format. Sur
    des découpes (grandes plages transparentes uniformes) le gain est énorme.

Référence : « Adobe Photoshop File Formats Specification », sections File
Header, Layer and Mask Information, Channel Image Data.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# Identifiants de canal du format : -1 = transparence, 0/1/2 = R/G/B.
_CHANNEL_IDS = (-1, 0, 1, 2)
_RGBA_INDEX = {-1: 3, 0: 0, 1: 1, 2: 2}

# Photoshop refuse au-delà (PSB sert au-dessus, on ne le gère pas).
MAX_SIDE = 30000


class PsdError(RuntimeError):
    pass


def _packbits(data: bytes) -> bytes:
    """Compression RLE PackBits, telle que la définit le format PSD.

    Écrite à la main parce que le `packbits` de la bibliothèque standard
    n'existe pas et que celui de numpy fait tout autre chose (bits, pas octets).
    """
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        # Série d'octets identiques (au moins 3) -> encodage par répétition.
        run = 1
        while (i + run < n and run < 128 and data[i + run] == data[i]):
            run += 1
        if run >= 3:
            out.append(257 - run)
            out.append(data[i])
            i += run
            continue
        # Sinon : bloc littéral, arrêté dès qu'une répétition de 3 pointe.
        start = i
        i += 1
        while i < n and (i - start) < 128:
            if (i + 2 < n and data[i] == data[i + 1] == data[i + 2]):
                break
            i += 1
        length = i - start
        out.append(length - 1)
        out += data[start:i]
    return bytes(out)


def _encode_channel(plane: np.ndarray) -> tuple[bytes, list[int]]:
    """Un canal en RLE : (données, longueur compressée de chaque ligne)."""
    rows = []
    counts = []
    for row in plane:
        enc = _packbits(row.tobytes())
        rows.append(enc)
        counts.append(len(enc))
    return b"".join(rows), counts


def _pascal_string(s: str, pad: int = 4) -> bytes:
    """Chaîne Pascal (longueur sur 1 octet) padée sur `pad`."""
    raw = (s or "layer").encode("latin-1", "replace")[:255]
    out = bytes([len(raw)]) + raw
    return out + b"\0" * ((-len(out)) % pad)


def _unicode_name(s: str) -> bytes:
    """Bloc « luni » : le VRAI nom du calque, en Unicode.

    Le nom Pascal du bloc principal est limité au latin-1 : sans « luni », un
    accent ou un emoji ressort en charabia dans Photoshop.
    """
    text = s or "layer"
    payload = struct.pack(">I", len(text)) + text.encode("utf-16-be")
    payload += b"\0" * ((-len(payload)) % 4)
    return b"8BIM" + b"luni" + struct.pack(">I", len(payload)) + payload


def _crop_to_content(rgba: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Recadre sur les pixels non transparents. (image, haut, gauche)."""
    alpha = rgba[:, :, 3]
    ys, xs = np.nonzero(alpha)
    if len(ys) == 0:
        return rgba[:1, :1], 0, 0        # calque vide : 1 px, pas une erreur
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    return rgba[top:bottom, left:right], top, left


def write_psd(path: Path | str, composite: np.ndarray,
              layers: list[tuple[str, np.ndarray]]) -> Path:
    """Écrit un PSD RGB 8 bits.

    `composite` : HxWx3 uint8, l'image aplatie (ce que voit un logiciel qui ne
    lit pas les calques — Photoshop l'utilise aussi comme aperçu).
    `layers` : liste (nom, RGBA HxWx4 uint8) **pleine toile**, du fond vers le
    premier plan. Le recadrage est fait ici : l'appelant n'a pas à s'en soucier.
    """
    path = Path(path)
    if composite.ndim != 3 or composite.shape[2] < 3:
        raise PsdError("The composite image must be RGB (HxWx3).")
    height, width = composite.shape[:2]
    if max(width, height) > MAX_SIDE:
        raise PsdError(f"PSD capped at {MAX_SIDE} px on a side "
                       f"(asked: {width}×{height}).")
    if not layers:
        raise PsdError("No layer to write.")

    buf = bytearray()
    # --- En-tête -----------------------------------------------------------
    buf += b"8BPS" + struct.pack(">H", 1) + b"\0" * 6
    buf += struct.pack(">HIIHH", 3, height, width, 8, 3)   # RGB 8 bits
    buf += struct.pack(">I", 0)          # données de mode colorimétrique
    buf += struct.pack(">I", 0)          # ressources image

    # --- Section calques ---------------------------------------------------
    records = bytearray()
    channel_blobs = bytearray()
    records += struct.pack(">h", len(layers))
    for name, rgba in layers:
        if rgba.shape[2] != 4:
            raise PsdError(f"Layer “{name}” must be RGBA.")
        crop, top, left = _crop_to_content(rgba)
        h, w = crop.shape[:2]
        records += struct.pack(">iiii", top, left, top + h, left + w)
        records += struct.pack(">H", len(_CHANNEL_IDS))

        encoded = []
        for cid in _CHANNEL_IDS:
            plane = np.ascontiguousarray(crop[:, :, _RGBA_INDEX[cid]])
            data, counts = _encode_channel(plane)
            # 2 octets de mode + 2 octets par ligne (table des longueurs).
            blob = struct.pack(">H", 1) + b"".join(
                struct.pack(">H", c) for c in counts) + data
            encoded.append(blob)
            records += struct.pack(">hI", cid, len(blob))

        records += b"8BIM" + b"norm"
        records += struct.pack(">BBBB", 255, 0, 0, 0)   # opacité, clip, flags
        extra = (struct.pack(">I", 0)          # masque du calque : aucun
                 + struct.pack(">I", 0)        # plages de fusion : aucune
                 + _pascal_string(name)
                 + _unicode_name(name))
        records += struct.pack(">I", len(extra)) + extra
        channel_blobs += b"".join(encoded)

    layer_info = bytes(records) + bytes(channel_blobs)
    if len(layer_info) % 2:
        layer_info += b"\0"
    section = struct.pack(">I", len(layer_info)) + layer_info
    section += struct.pack(">I", 0)      # informations de masque global
    buf += struct.pack(">I", len(section)) + section

    # --- Image composite ---------------------------------------------------
    # Même encodage RLE : c'est la moitié du fichier sur une image lisse.
    comp_rows, comp_counts = [], []
    for c in range(3):
        data, counts = _encode_channel(
            np.ascontiguousarray(composite[:, :, c]))
        comp_rows.append(data)
        comp_counts += counts
    buf += struct.pack(">H", 1)
    buf += b"".join(struct.pack(">H", c) for c in comp_counts)
    buf += b"".join(comp_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path
