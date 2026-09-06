"""Interface language — English only.

The application used to carry French source strings plus an English
translation table. The strings are now written in English directly, so there
is nothing left to translate: `t()` returns what it is given.

The module is kept — rather than deleted and its ~40 call sites edited —
because it is the seam where a language layer would go back if one is ever
wanted again. Everything here is deliberately an identity function; nothing
imports a dictionary any more.
"""
from __future__ import annotations

_LANG = "en"


def set_lang(lang: str) -> None:      # noqa: ARG001 - signature conservée
    """Accepted and ignored: the interface only exists in English."""


def get_lang() -> str:
    return _LANG


def init_from_prefs() -> str:
    return _LANG


def t(s):
    """The string itself. Kept so call sites read the same as before."""
    return s


def to_source(s):
    """Menu choices are their own key now — identity."""
    return s


def translate_blocks(demo) -> None:   # noqa: ARG001 - signature conservée
    """Nothing to translate after the interface is built."""
