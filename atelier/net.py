"""Utilitaires réseau : adresses IP locales et recherche de port libre.

Sert à exposer Atelier sur le réseau local (collègues sur le même Wi-Fi/LAN).
"""
from __future__ import annotations

import socket


def primary_ip() -> str:
    """IP locale principale (interface utilisée pour sortir vers le réseau)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def lan_ips() -> list[str]:
    """Toutes les IPv4 locales plausibles (hors loopback)."""
    ips = {primary_ip()}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def find_free_port(start: int, count: int = 25, host: str = "") -> int:
    """Premier port libre à partir de `start`. host='' = toutes interfaces."""
    bind_host = "" if host in ("0.0.0.0", "") else host
    for p in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((bind_host, p))
                return p
            except OSError:
                continue
    return start
