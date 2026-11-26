"""Shared data model for the whole pipeline.

Every extracted value is a `Field` that carries the value AND whether it was
read clearly. Anything not read clearly is rendered as "CHECK" in outputs -
we never guess silently.
"""
from __future__ import annotations

from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field as PydField

T = TypeVar("T")

CHECK = "CHECK"


class Confidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Field(BaseModel, Generic[T]):
    """A single extracted value.

    value: the value read from the drawing, or None if not present/unreadable.
    clear: True only if the value was read unambiguously from the drawing.
    shown: False means the drawing simply doesn't show this (not an error).

    The three states that matter (the contract between extract and validate):
      written & legible     -> value set,  clear=True,  shown=True
      written but unclear   -> value set or None, clear=False, shown=True
      not written at all    -> value None, clear=False, shown=False
    Anything derived/inferred (not read) must be clear=False.
    """

    value: Optional[T] = None
    clear: bool = False
    shown: bool = True

    def display(self, fmt=str, required: bool = False) -> str:
        """Render for Excel/PDF: blank if not shown (unless required), CHECK if unclear."""
        if not self.shown and self.value is None and not required:
            return ""
        if self.value is None or not self.clear:
            return CHECK if self.value is None else f"{fmt(self.value)} ({CHECK})"
        return fmt(self.value)


class Hole(BaseModel):
    diameter_mm: Field[float] = PydField(default_factory=Field)
    # Free text position as written/drawn, e.g. "50 from left, 100 from top"
    position: Field[str] = PydField(default_factory=lambda: Field(shown=False))


class Feature(BaseModel):
    """Notch, offcut, groove or edge work."""

    kind: str  # "notch" | "offcut" | "groove" | "edge work" | other
    size: Field[str] = PydField(default_factory=lambda: Field(shown=False))
    position: Field[str] = PydField(default_factory=lambda: Field(shown=False))
    note: str = ""


class Panel(BaseModel):
    panel_id: str
    width_mm: Field[float] = PydField(default_factory=Field)
    height_mm: Field[float] = PydField(default_factory=Field)
    quantity: Field[int] = PydField(default_factory=lambda: Field(value=1, clear=True, shown=False))
    thickness_mm: Field[float] = PydField(default_factory=lambda: Field(shown=False))
    glass_type: Field[str] = PydField(default_factory=lambda: Field(shown=False))
    holes: list[Hole] = PydField(default_factory=list)
    notches_offcuts: list[Feature] = PydField(default_factory=list)
    grooves_edgework: list[Feature] = PydField(default_factory=list)
    notes: str = ""
    # Problems noticed during extraction (e.g. implausible hole count) that
    # validate() turns into hard flags:
    extraction_issues: list[str] = PydField(default_factory=list)
    # Filled by validation:
    confidence: Confidence = Confidence.HIGH
    flags: list[str] = PydField(default_factory=list)

    @property
    def needs_check(self) -> bool:
        return bool(self.flags) or self.confidence != Confidence.HIGH

    @property
    def area_uncertain(self) -> bool:
        """True if the area rests on any value that isn't clearly read."""
        q = self.quantity
        qty_bad = (q.shown or q.value is not None) and (q.value is None or not q.clear)
        return (not self.width_mm.clear or not self.height_mm.clear
                or self.width_mm.value is None or self.height_mm.value is None or qty_bad)

    @property
    def hole_count_uncertain(self) -> bool:
        """A 'hole' with no diameter at all may be a fitting mark or symbol, not
        a drilled hole - the count needs a human (a written-but-unclear
        diameter is still a hole, so it doesn't count here)."""
        return any(h.diameter_mm.value is None for h in self.holes)

    @property
    def area_m2(self) -> Optional[float]:
        """Area for ALL pieces on this line (width x height x qty), m^2."""
        w, h, q = self.width_mm.value, self.height_mm.value, self.quantity.value or 1
        if w is None or h is None:
            return None
        return round(w * h * q / 1_000_000, 3)


class Drawing(BaseModel):
    """Everything extracted from one photo."""

    overall_width_mm: Field[float] = PydField(default_factory=lambda: Field(shown=False))
    overall_height_mm: Field[float] = PydField(default_factory=lambda: Field(shown=False))
    panels: list[Panel] = PydField(default_factory=list)
    general_notes: str = ""
    units_written: str = "mm"   # mm / cm / m / inch / unknown / none (not written)
    units_converted_from: str = ""  # set by units.convert_to_mm, e.g. "cm (assumed - no unit written)"
    model_used: str = ""        # which Gemini model read the drawing
    extraction_issues: list[str] = PydField(default_factory=list)
    # Drawing-level flags from validation (e.g. widths don't sum to overall)
    flags: list[str] = PydField(default_factory=list)


class LPO(BaseModel):
    lpo_no: str
    date: str  # ISO yyyy-mm-dd
    source_image: str
    drawing: Drawing

    @property
    def total_area_m2(self) -> float:
        return round(sum(p.area_m2 or 0 for p in self.drawing.panels), 3)

    @property
    def uncertain_area_lines(self) -> int:
        """Lines whose area is missing or rests on unclear values (total is then CHECK)."""
        return sum(1 for p in self.drawing.panels if p.area_uncertain)

    @property
    def total_holes(self) -> int:
        return sum(len(p.holes) * (p.quantity.value or 1) for p in self.drawing.panels)

    def all_flags(self) -> list[str]:
        out = list(self.drawing.flags)
        for p in self.drawing.panels:
            out += [f"{p.panel_id}: {f}" for f in p.flags]
        return out


EXCEL_COLUMNS = [
    "LPO No", "Line", "Panel ID", "Width (mm)", "Height (mm)", "Qty",
    "Thickness (mm)", "Glass type", "Area (m²)", "Hole count",
    "Hole diameters (mm)", "Hole positions", "Notches/offcuts",
    "Grooves/edge work", "Notes", "Confidence", "Needs check (Y/N)",
]
