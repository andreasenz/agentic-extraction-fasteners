from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from mirascope import llm
from mirascope.llm.providers import OllamaProvider
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm

# -----------------------------------------------------------------------------
# MODEL SETUP
# Mirascope v2 uses provider registration + prefixed model IDs.
# Register Ollama so any model ID like "ollama/<name>" is routed to it.
# MODEL_NAME should be the bare Ollama model name, e.g. "qwen3:8b" or "llava".
# -----------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME")
print(f"Using model: {MODEL_NAME}")

llm.register_provider(
    OllamaProvider(base_url="http://localhost:11434/v1"),
    scope="ollama/",
)
MODEL_ID = f"ollama/{MODEL_NAME}"


# -----------------------------------------------------------------------------
# ENUMS
# -----------------------------------------------------------------------------

class FastenerFamily(str, Enum):
    BOLT = "bolt"
    NUT = "nut"
    RIVET = "rivet"
    WASHER = "washer"
    SCREW = "screw"
    STUD = "stud"
    PIN = "pin"
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# SHARED BUILDING BLOCKS
# -----------------------------------------------------------------------------

class Tolerance(BaseModel):
    upper: Optional[float] = Field(default=None, description="Upper deviation")
    lower: Optional[float] = Field(default=None, description="Lower deviation")
    iso_class: Optional[str] = Field(default=None, description="ISO fit/tolerance class")


class DimVal(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = Field(default=None, description="'mm' or 'in'")
    tolerance: Optional[Tolerance] = None


class ThreadSpec(BaseModel):
    designation: Optional[str] = Field(default=None, description="e.g. M10x1.5, 3/8-16 UNC")
    standard: Optional[str] = Field(default=None, description="ISO, ANSI, DIN, BSP …")
    pitch_mm: Optional[float] = None
    thread_class: Optional[str] = Field(default=None, description="e.g. 6g, 2A")
    hand: Optional[str] = Field(default=None, description="RH or LH")
    spacing: Optional[str] = None
    thread_type: Optional[str] = None
    fit: Optional[str] = None


class MaterialSpec(BaseModel):
    base_material: Optional[str] = None
    grade_or_standard: Optional[str] = None
    hardness: Optional[str] = None
    tensile_strength: Optional[str] = None
    surface_treatment: Optional[str] = None
    finish: Optional[str] = None


class DrawingMeta(BaseModel):
    drawing_number: Optional[str] = None
    revision: Optional[str] = None
    title: Optional[str] = None
    applicable_standard: Optional[str] = None
    specifications_met: Optional[str] = None
    system_of_measurement: Optional[str] = None
    country_of_origin: Optional[str] = None


class ComplianceSpec(BaseModel):
    dfars_compliance: Optional[str] = None
    eccn: Optional[str] = None
    reach_compliance: Optional[str] = None
    rohs_compliance: Optional[str] = None
    schedule_b_number: Optional[str] = None
    usmca_qualifying: Optional[str] = None
    certificate_type: Optional[str] = None
    certificate_form: Optional[str] = None


class Annotation(BaseModel):
    label: Optional[str] = None
    text: Optional[str] = None


class BaseFastenerExtraction(BaseModel):
    family: FastenerFamily
    meta: Optional[DrawingMeta] = None
    material: Optional[MaterialSpec] = None
    compliance: Optional[ComplianceSpec] = None
    annotations: list[Annotation] = Field(default_factory=list)
    confidence_notes: Optional[str] = None
    raw_attributes: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# FAMILY-SPECIFIC MODELS
# -----------------------------------------------------------------------------

class BoltExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.BOLT
    thread: Optional[ThreadSpec] = None
    nominal_diameter: Optional[DimVal] = None
    length: Optional[DimVal] = Field(default=None, description="Measured under head")
    thread_length: Optional[DimVal] = None
    head_type: Optional[str] = None
    head_height: Optional[DimVal] = None
    head_width_across_flats: Optional[DimVal] = None
    drive_type: Optional[str] = None
    point_type: Optional[str] = None
    property_class: Optional[str] = None


class NutExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.NUT
    thread: Optional[ThreadSpec] = None
    nominal_diameter: Optional[DimVal] = None
    nut_type: Optional[str] = None
    profile: Optional[str] = None
    height: Optional[DimVal] = None
    width_across_flats: Optional[DimVal] = None
    width_across_corners: Optional[DimVal] = None
    chamfer_angle: Optional[float] = None
    property_class: Optional[str] = None
    strength_rating: Optional[str] = None
    prevailing_torque_feature: Optional[str] = None
    drive_style: Optional[str] = None


class RivetExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.RIVET
    rivet_type: Optional[str] = None
    nominal_diameter: Optional[DimVal] = None
    length: Optional[DimVal] = None
    max_grip: Optional[DimVal] = None
    min_grip: Optional[DimVal] = None
    head_style: Optional[str] = None
    head_diameter: Optional[DimVal] = None
    head_height: Optional[DimVal] = None
    set_head_type: Optional[str] = None
    drill_diameter: Optional[DimVal] = None
    mandrel_material: Optional[str] = None


class WasherExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.WASHER
    washer_type: Optional[str] = None
    inner_diameter: Optional[DimVal] = None
    outer_diameter: Optional[DimVal] = None
    thickness: Optional[DimVal] = None
    inner_diameter_tolerance: Optional[Tolerance] = None
    outer_diameter_tolerance: Optional[Tolerance] = None
    hardness: Optional[str] = None
    applicable_bolt_size: Optional[str] = None


class ScrewExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.SCREW
    thread: Optional[ThreadSpec] = None
    nominal_diameter: Optional[DimVal] = None
    length: Optional[DimVal] = Field(default=None, description="Measured under head")
    head_type: Optional[str] = None
    head_profile: Optional[str] = None
    head_diameter: Optional[DimVal] = None
    head_height: Optional[DimVal] = None
    drive_type: Optional[str] = None
    drive_size: Optional[str] = None
    point_type: Optional[str] = None
    threading: Optional[str] = None
    property_class: Optional[str] = None
    performance: Optional[str] = None
    decimal_size_equivalent: Optional[str] = None


class StudExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.STUD
    thread_plant_end: Optional[ThreadSpec] = None
    thread_nut_end: Optional[ThreadSpec] = None
    nominal_diameter: Optional[DimVal] = None
    overall_length: Optional[DimVal] = None
    plant_end_length: Optional[DimVal] = None
    nut_end_length: Optional[DimVal] = None
    unthreaded_shank_length: Optional[DimVal] = None
    property_class: Optional[str] = None


class PinExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.PIN
    pin_type: Optional[str] = None
    nominal_diameter: Optional[DimVal] = None
    length: Optional[DimVal] = None
    taper: Optional[str] = None
    chamfer: Optional[DimVal] = None
    fit: Optional[str] = None


# -----------------------------------------------------------------------------
# UNKNOWN / TEMPLATE PROPOSAL MODELS
# -----------------------------------------------------------------------------

class TemplateFieldProposal(BaseModel):
    name: str
    field_type: str
    description: Optional[str] = None
    required: bool = False


class TemplateProposal(BaseModel):
    family_name: str
    base_class: str = "BaseFastenerExtraction"
    suggested_fields: list[TemplateFieldProposal] = Field(default_factory=list)
    sample_output: dict[str, Any] = Field(default_factory=dict)


class UnknownFastenerExtraction(BaseFastenerExtraction):
    family: FastenerFamily = FastenerFamily.UNKNOWN
    detected_name: Optional[str] = None
    distinguishing_features: list[str] = Field(default_factory=list)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    template_proposal: Optional[TemplateProposal] = None


class FamilyDecision(BaseModel):
    family: FastenerFamily
    custom_family_name: Optional[str] = None
    is_supported: bool
    reasoning: Optional[str] = None


AnyFastenerExtraction = (
    BoltExtraction
    | NutExtraction
    | RivetExtraction
    | WasherExtraction
    | ScrewExtraction
    | StudExtraction
    | PinExtraction
    | UnknownFastenerExtraction
)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def resize_image(path: Path, max_size: int = 1024) -> bytes:
    img = Image.open(path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def to_pascal_case(name: str) -> str:
    parts = [p for p in name.replace("-", "_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def normalize_identifier(name: str) -> str:
    out = []
    for c in name.lower().strip():
        out.append(c if c.isalnum() else "_")
    result = "".join(out)
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_") or "custom_fastener"


def generate_pydantic_model_code(template: TemplateProposal) -> str:
    class_name = f"{to_pascal_case(normalize_identifier(template.family_name))}Extraction"
    lines = [
        "from __future__ import annotations",
        "",
        "from typing import Optional, Any",
        "from pydantic import Field",
        "",
        "# Assumes BaseFastenerExtraction and FastenerFamily are already defined/imported",
        f"class {class_name}(BaseFastenerExtraction):",
        "    family: FastenerFamily = FastenerFamily.UNKNOWN",
    ]
    if not template.suggested_fields:
        lines.append("    raw_attributes: dict[str, Any] = Field(default_factory=dict)")
        return "\n".join(lines)
    for field in template.suggested_fields:
        field_name = normalize_identifier(field.name)
        field_type = field.field_type or "str"
        if field.required:
            lines.append(f"    {field_name}: {field_type}")
        else:
            lines.append(f"    {field_name}: Optional[{field_type}] = None")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# PROMPTS
# -----------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "You are a mechanical fastener classification specialist. "
    "Look at the engineering drawing image and classify the part. "
    "Return a JSON object with fields: family, custom_family_name, is_supported, reasoning. "
    "Supported families are exactly: bolt, nut, rivet, washer, screw, stud, pin. "
    "If the object belongs to one of these, set is_supported=true and set family accordingly. "
    "If it is a fastener but not in the supported list, set family='unknown', is_supported=false, "
    "and provide a short custom_family_name such as retaining_ring, circlip, anchor, insert, clamp, etc."
)

_EXTRACTION_SYSTEM = (
    "You are an expert fastener analyst. "
    "Return only valid JSON matching the target schema exactly. "
    "Do not flatten nested objects. "
    "Include optional fields with null if not visible. "
    "Use nested objects exactly when present: meta, thread, material, compliance. "
    "Example for thread: {\"designation\": \"M10x1.5\", \"pitch_mm\": 1.5, \"hand\": \"RH\"}. "
    "Use null for unknown values. "
    "Do not invent top-level keys not in the schema. "
    "Store extra attributes under raw_attributes. "
    "Store standards/metadata in meta or compliance. "
    "Explain ambiguities in confidence_notes."
)

_UNKNOWN_EXTRACTION_SYSTEM = (
    "You are an expert fastener analyst. "
    "The fastener may not belong to a supported family. "
    "Return only valid JSON matching the UnknownFastenerExtraction schema exactly. "
    "Extract all visible attributes into extracted_fields. "
    "List distinguishing_features. "
    "Propose a template_proposal with a concise family_name and suggested fields. "
    "Use simple field types: str, float, DimVal, ThreadSpec, MaterialSpec, bool. "
    "Put a realistic example in sample_output."
)

_PROMPTS: dict[FastenerFamily, str] = {
    FastenerFamily.BOLT: (
        "Extract all visible bolt information: thread designation, nominal diameter, length, "
        "thread length, head type, head height, width across flats, drive type, point type, "
        "property class, material, finish, standards, certifications, notes."
    ),
    FastenerFamily.NUT: (
        "Extract all visible nut information: nut type, profile, thread designation, spacing, "
        "fit, direction, nominal diameter, height, width across flats, width across corners, "
        "chamfer angle, material, strength rating, property class, prevailing torque features, "
        "drive style, specifications, certificates, compliance, notes."
    ),
    FastenerFamily.RIVET: (
        "Extract all visible rivet information: rivet type, nominal diameter, length, grip range, "
        "head style, head dimensions, set head type, drill diameter, mandrel material, material, "
        "finish, standards, compliance, notes."
    ),
    FastenerFamily.WASHER: (
        "Extract all visible washer information: washer type, inner diameter, outer diameter, "
        "thickness, tolerances, hardness, material, applicable bolt size, standards, compliance, notes."
    ),
    FastenerFamily.SCREW: (
        "Extract all visible screw information: thread designation, spacing, thread type, fit, "
        "direction, nominal diameter, length, head type, head profile, head diameter, head height, "
        "drive type, drive size, point type, threading, material, hardness, tensile strength, "
        "property class, performance, standards, compliance, notes. "
        "IMPORTANT: populate thread.pitch_mm with the numeric pitch in mm (e.g. 1.5 for M10x1.5). "
        "Populate thread.designation with the exact string (e.g. 'M10x1.5' or '1/4-20')."
    ),
    FastenerFamily.STUD: (
        "Extract all visible stud information: both thread ends, nominal diameter, overall length, "
        "each threaded length, unthreaded shank length, material, property class, standards, compliance, notes."
    ),
    FastenerFamily.PIN: (
        "Extract all visible pin information: pin type, nominal diameter, length, taper, chamfer, "
        "fit, material, standards, compliance, notes."
    ),
}

_SCHEMA_MAP: dict[FastenerFamily, type] = {
    FastenerFamily.BOLT: BoltExtraction,
    FastenerFamily.NUT: NutExtraction,
    FastenerFamily.RIVET: RivetExtraction,
    FastenerFamily.WASHER: WasherExtraction,
    FastenerFamily.SCREW: ScrewExtraction,
    FastenerFamily.STUD: StudExtraction,
    FastenerFamily.PIN: PinExtraction,
}


# -----------------------------------------------------------------------------
# AGENTS
# Mirascope v2 uses @llm.call with a model ID and an optional format= for
# structured output. format=llm.format(Model, mode="json") uses JSON-mode,
# which is what Ollama supports. Each decorated async function returns an
# AsyncResponse; call response.parse() to get the typed Pydantic instance.
#
# The factory _make_extractor creates one typed extractor per fastener family.
# The model ID is captured in the closure at factory-call time so the decorator
# sees the correct string even though MODEL_ID is computed at module level.
# -----------------------------------------------------------------------------

@llm.call(MODEL_ID, format=llm.format(FamilyDecision, mode="json"), temperature=0.0)
async def _classify_fastener(image_b64: str) -> list:
    return [
        llm.SystemMessage(content=llm.Text(text=_CLASSIFIER_SYSTEM)),
        llm.UserMessage(content=[
            llm.Image(source=llm.Base64ImageSource(type="base64_image_source", data=image_b64, mime_type="image/png")),
            llm.Text(text="Classify the fastener in this engineering drawing."),
        ]),
    ]


def _make_extractor(schema_class: type) -> Any:
    model_id = MODEL_ID  # capture current value into closure

    @llm.call(model_id, format=llm.format(schema_class, mode="json"), temperature=0.0)
    async def _extract(image_b64: str, system: str, prompt: str) -> list:
        return [
            llm.SystemMessage(content=llm.Text(text=system)),
            llm.UserMessage(content=[
                llm.Image(source=llm.Base64ImageSource(type="base64_image_source", data=image_b64, mime_type="image/png")),
                llm.Text(text=prompt),
            ]),
        ]
    return _extract


_EXTRACTORS: dict[FastenerFamily, Any] = {
    family: _make_extractor(schema_class)
    for family, schema_class in _SCHEMA_MAP.items()
}

_UNKNOWN_EXTRACTOR = _make_extractor(UnknownFastenerExtraction)


# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------

async def process_image(image_path: Path):
    img_bytes = resize_image(image_path)
    image_b64 = to_b64(img_bytes)
    print(f"  Image: {len(img_bytes) / 1024:.1f} KB")

    print("  [1/2] Classifying …")
    cls_response = await _classify_fastener(image_b64)
    decision: FamilyDecision = cls_response.parse()
    print(f"  [1/2] Detected → family={decision.family.value}, supported={decision.is_supported}")

    print("  [2/2] Extracting …")
    if decision.is_supported and decision.family in _EXTRACTORS:
        extractor = _EXTRACTORS[decision.family]
        system = _EXTRACTION_SYSTEM
        prompt = _PROMPTS[decision.family]
    else:
        extractor = _UNKNOWN_EXTRACTOR
        system = _UNKNOWN_EXTRACTION_SYSTEM
        custom_name = decision.custom_family_name or "unknown_fastener"
        prompt = (
            f"This fastener is not in the supported templates. "
            f"Proposed family name: {custom_name}. "
            f"Extract all visible information and propose a new template."
        )

    MAX_RETRIES = 3
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            ext_response = await extractor(image_b64, system, prompt)
            extraction = ext_response.parse()
            print("  [2/2] Done ✓")
            return decision, extraction
        except Exception as e:
            last_error = e
            print(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            continue
    raise last_error


async def process_directory(input_dir: Path, output_dir: Path):
    pngs = sorted(input_dir.glob("*.png"))
    if not pngs:
        print("No PNG files found.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = output_dir / "proposed_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for png in tqdm(pngs):
        print(f"\n📐 {png.name}")
        exists = any(f.startswith(png.stem) for f in os.listdir(output_dir))
        if exists:
            print("  File already processed")
            continue

        try:
            decision, extraction = await process_image(png)
            results.append((png, decision, extraction))

            family_name = (
                decision.family.value if decision.is_supported
                else normalize_identifier(decision.custom_family_name or "unknown_fastener")
            )
            out_json = output_dir / f"{png.stem}__{family_name}.json"
            out_json.write_text(
                extraction.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8",
            )

            if isinstance(extraction, UnknownFastenerExtraction) and extraction.template_proposal:
                code = generate_pydantic_model_code(extraction.template_proposal)
                out_py = template_dir / f"{normalize_identifier(extraction.template_proposal.family_name)}.py"
                out_py.write_text(code, encoding="utf-8")
                print(f"  🧩 Template → {out_py.name}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    return results


async def main():
    if not MODEL_NAME:
        raise RuntimeError("Set the MODEL_NAME environment variable before running.")

    if len(sys.argv) < 2:
        print("Usage: python mirascope_pipeline.py <image.png|directory> [output_dir]")
        sys.exit(1)

    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("extractions")

    if not src.exists():
        raise FileNotFoundError(f"Path not found: {src}")

    if src.is_dir():
        results = await process_directory(src, out)
        print(f"\n✅ Processed {len(results)} drawing(s) → {out}/")
    else:
        decision, extraction = await process_image(src)
        out.mkdir(parents=True, exist_ok=True)

        family_name = (
            decision.family.value if decision.is_supported
            else normalize_identifier(decision.custom_family_name or "unknown_fastener")
        )
        out_file = out / f"{src.stem}__{family_name}.json"
        out_file.write_text(
            extraction.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        print(f"\nExtraction saved → {out_file}")
        print(extraction.model_dump_json(indent=2, exclude_none=True))

        if isinstance(extraction, UnknownFastenerExtraction) and extraction.template_proposal:
            template_dir = out / "proposed_templates"
            template_dir.mkdir(parents=True, exist_ok=True)
            code = generate_pydantic_model_code(extraction.template_proposal)
            out_py = template_dir / f"{normalize_identifier(extraction.template_proposal.family_name)}.py"
            out_py.write_text(code, encoding="utf-8")
            print(f"\nProposed template → {out_py}")


if __name__ == "__main__":
    asyncio.run(main())
