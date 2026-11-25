"""A fixed glass-type vocabulary.

Hand drawings describe the same glass many ways: "14mm UNIT", "DGU", "IGU",
"double glazed" are all a double-glazed unit; "tempered" is "toughened".
`canonical_glass_type` maps free text onto fixed words in a fixed order:

    [tint / finish]  [toughened]  [laminated | wired | double glazed unit]

    "10mm Clear Toughened"   -> "clear toughened"
    "14mm UNIT"              -> "double glazed unit"
    "bronze tinted tempered" -> "tinted bronze toughened"
    "glass" / "glazing"      -> None   (marks a glazed area, not a type)
    "KITCHEN SPLASHBACK"     -> None   (not a glass description)

Thickness is not part of the type (it has its own column).
"""
from __future__ import annotations

import re
from typing import Optional


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# One tint/colour at most; first match wins (most specific first).
_TINTS = [
    ("tinted bronze", _rx(r"bronze")),
    ("tinted grey", _rx(r"\bgr[ae]y\b|smoke[d]?\b")),
    ("tinted green", _rx(r"\bgreen\b")),
    ("tinted blue", _rx(r"\bblue\b")),
    ("tinted", _rx(r"\btint")),
]
# Finishes / coatings; several may apply.
_FINISHES = [
    ("extra clear", _rx(r"extra[- ]?clear|low[- ]?iron|starphire|optiwhite")),
    ("clear", _rx(r"\bclear\b|\bfloat\b|\bannealed\b")),
    ("frosted", _rx(r"frost|sandblast|acid[- ]?etch|\betched\b|obscure|\bsatin\b|privacy|\bopal\b")),
    ("patterned", _rx(r"pattern|fluted|reeded")),
    ("reflective", _rx(r"reflect|\bsolar\b")),
    ("low-e", _rx(r"low[- ]?e\b")),
    ("mirror", _rx(r"mirror")),
    ("back painted", _rx(r"back[- ]?painted|lacquer|\bpainted\b")),
]
_TOUGHENED = _rx(r"tough|temper|\bsafety\b")
_CONSTRUCTION = [
    ("double glazed unit", _rx(r"\bdgu\b|\bigu\b|\bunits?\b|double[- ]?glaz|insulat|sealed")),
    ("laminated", _rx(r"laminat|\blam\b")),
    ("wired", _rx(r"\bwired?\b")),
]
# Words that only say "glass goes here".
_GENERIC = _rx(r"\bglass\b|\bglaz(ed|ing)\b|\bpanes?\b")


def canonical_glass_type(text: Optional[str]) -> Optional[str]:
    """Fixed-vocabulary glass type for `text`, or None if it describes no glass."""
    t = text or ""
    parts: list[str] = []
    for name, rx in _TINTS:
        if rx.search(t):
            parts.append(name)
            break
    for name, rx in _FINISHES:
        if rx.search(t) and not (name == "clear" and "extra clear" in parts):
            parts.append(name)
    if _TOUGHENED.search(t):
        parts.append("toughened")
    for name, rx in _CONSTRUCTION:
        if rx.search(t):
            parts.append(name)
    return " ".join(parts) or None


def is_generic_glass_word(text: Optional[str]) -> bool:
    """True if the text only says 'glass' (a glazed area / title), no type."""
    return canonical_glass_type(text) is None and bool(_GENERIC.search(text or ""))
