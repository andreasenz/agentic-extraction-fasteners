
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool


import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, BinaryContent
from tqdm import tqdm
import os
import re

import logging
import httpx


# -----------------------------------------------------------------------------
# MODEL SETUP
# -----------------------------------------------------------------------------
# Sostituisci questa parte con il model/provider che stai usando davvero.
#
# Esempio:
# from pydantic_ai.models.openai import OpenAIChatModel
# from pydantic_ai.providers.ollama import OllamaProvider
#
# model = OpenAIChatModel(
#     "qwen3.5:0.8b",
#     provider=OllamaProvider(base_url="http://localhost:11434/v1")
# )
#
# Oppure usa il tuo oggetto `model` già creato prima di questo file.
# -----------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME")
print(MODEL_NAME)
exit
# model = ...
model = OpenAIChatModel(
        MODEL_NAME,
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    

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
    designation: str = Field(default=None, description="e.g. M10x1.5, 3/8-16 UNC")
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
    length: DimVal = Field(default=None, description="Measured under head")
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
    height: DimVal = None
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
    length: DimVal = None
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
from PIL import Image
import io

def resize_image(path: Path, max_size: int = 512) -> bytes:
    img = Image.open(path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def image_message(path: Path, prompt: str):
    return [
        BinaryContent(data=path.read_bytes(), media_type="image/png"),
        prompt,
    ]

def to_pascal_case(name: str) -> str:
    parts = [p for p in name.replace("-", "_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def normalize_identifier(name: str) -> str:
    out = []
    for c in name.lower().strip():
        if c.isalnum():
            out.append(c)
        else:
            out.append("_")
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
    "You MUST respond with valid JSON only. No other text. "
    "Look at the engineering drawing image and classify the part. "
    "Return a JSON object with fields: family, custom_family_name, is_supported, reasoning. "
    "Supported families are exactly: bolt, nut, rivet, washer, screw, stud, pin. "
    "If the object belongs to one of these, set is_supported=true and set family accordingly. "
    "If it is a fastener but not in the supported list, set family='unknown', is_supported=false, "
    "and provide a short custom_family_name such as retaining_ring, circlip, anchor, insert, clamp, etc. "
    "Return only valid JSON."
)

_EXTRACTION_SYSTEM = (
    "You are an expert fastener analyst. "
    "You MUST respond with valid JSON only. No other text. "
    "Return only valid JSON matching the target schema exactly. "
    "Do not flatten nested objects. "
    "Even if some fields are optional, they are important for the structure of the data, so include them with null if not visible. "
    "Use nested objects exactly when present, such as meta, thread, material, compliance. "
    "Example for thread: {\"designation\": \"M10x1.5\", \"pitch_mm\": 1.5, \"hand\": \"RH\"}. "
    "Use null for unknown values. "
    "Do not invent top-level keys not present in the schema. "
    "If you find useful attributes that do not fit cleanly, store them under raw_attributes. "
    "If standards or commercial metadata are visible, store them in meta or compliance where possible. "
    "If something is ambiguous, explain it briefly in confidence_notes."
    #"If you cannot assign an extracted attribute to a field, search the web for common attributes of that fastener type and see if any fit the observed data. Search for UNC, ISO or DIN standards, or common fastener features. Use the duckduckgo_search_tool for this. Use the output from the tool call and think step by step to see if any of the attributes you find can be matched to the observed data and schema fields. If you find a good match, populate the relevant field and cite the source in confidence_notes."

)

_UNKNOWN_EXTRACTION_SYSTEM = (
    "You are an expert fastener analyst. "
    "You MUST respond with valid JSON only. No other text. "
    "The fastener may not belong to a supported family. "
    "Return only valid JSON matching the UnknownFastenerExtraction schema exactly. "
    "Extract all visible attributes into extracted_fields. "
    "List distinguishing_features. "
    "Then propose a template_proposal with a concise family_name and a list of suggested fields. "
    "Use simple field types like str, float, DimVal, ThreadSpec, MaterialSpec, bool. "
    "Put a realistic example in sample_output."
)

_PROMPTS: dict[FastenerFamily, str] = {
    FastenerFamily.BOLT: (
        "Extract all visible bolt information. "
        "Look for thread designation, nominal diameter, length, thread length, head type, "
        "head height, width across flats, drive type, point type, property class, material, "
        "finish, standards, certifications, and notes."
    ),
    FastenerFamily.NUT: (
        "Extract all visible nut information. "
        "Look for nut type, profile, thread designation, thread spacing, thread fit, thread direction, "
        "nominal diameter, height, width across flats, width across corners, chamfer angle, "
        "material, strength rating, property class, prevailing torque features, drive style, "
        "specifications met, certificates, compliance, and notes."
    ),
    FastenerFamily.RIVET: (
        "Extract all visible rivet information. "
        "Look for rivet type, nominal diameter, length, grip range, head style, head dimensions, "
        "set head type, drill diameter, mandrel material, material, finish, standards, compliance, and notes."
    ),
    FastenerFamily.WASHER: (
        "Extract all visible washer information. "
        "Look for washer type, inner diameter, outer diameter, thickness, tolerances, hardness, "
        "material, applicable bolt size, standards, compliance, and notes."
    ),
    FastenerFamily.SCREW: (
    "Extract all visible screw information. "
    "Look for thread designation, spacing, thread type, fit, direction, nominal diameter, length, "
    "head type, head profile, head diameter, head height, drive type, drive size, point type, threading, "
    "material, hardness, tensile strength, property class, performance, standards, compliance, and notes. "
    "IMPORTANT: for thread pitch, populate thread.pitch_mm with the numeric pitch value in millimetres "
    "(e.g. for M10x1.5 set pitch_mm=1.5). "
    "If the drawing shows the thread designation as a string like 'M10x1.5' or '1/4-20', "
    "also populate thread.designation with that exact string."
),
    FastenerFamily.STUD: (
        "Extract all visible stud information. "
        "Look for both thread ends, nominal diameter, overall length, each threaded length, "
        "unthreaded shank length, material, property class, standards, compliance, and notes."
    ),
    FastenerFamily.PIN: (
        "Extract all visible pin information. "
        "Look for pin type, nominal diameter, length, taper, chamfer, fit, material, "
        "standards, compliance, and notes."
    ),
}


# -----------------------------------------------------------------------------
# AGENTS
# -----------------------------------------------------------------------------
classifier_agent: Agent[None, FamilyDecision] = Agent(
    model=model,
    system_prompt=_CLASSIFIER_SYSTEM,
    model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}
)
"""
_AGENTS: dict[FastenerFamily, Agent] = {
    FastenerFamily.BOLT: Agent(model, tools=[duckduckgo_search_tool()], output_type=BoltExtraction, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0}),
    FastenerFamily.NUT: Agent(model, tools=[duckduckgo_search_tool()], output_type=NutExtraction, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0}),
    FastenerFamily.RIVET: Agent(model, tools=[duckduckgo_search_tool()], output_type=RivetExtraction, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0}),
    FastenerFamily.WASHER: Agent(model, tools=[duckduckgo_search_tool()], output_type=WasherExtraction, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0}),
    FastenerFamily.SCREW: Agent(model, tools=[duckduckgo_search_tool()], output_type=ScrewExtraction, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0}),
    FastenerFamily.STUD: Agent(model, tools=[duckduckgo_search_tool()], output_type=StudExtraction, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0}),
    FastenerFamily.PIN: Agent(model, tools=[duckduckgo_search_tool()], output_type=PinExtraction, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0}),
}"""
_AGENTS: dict[FastenerFamily, Agent] = {
    FastenerFamily.BOLT: Agent(model, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}),
    FastenerFamily.NUT: Agent(model, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}),
    FastenerFamily.RIVET: Agent(model, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}),
    FastenerFamily.WASHER: Agent(model, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}),
    FastenerFamily.SCREW: Agent(model, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}),
    FastenerFamily.STUD: Agent(model, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}),
    FastenerFamily.PIN: Agent(model, system_prompt=_EXTRACTION_SYSTEM, model_settings={"temperature": 0.0, "extra_body": {"format": "json"}}),
}
unknown_agent: Agent[None, UnknownFastenerExtraction] = Agent(
    model=model,
    output_type=UnknownFastenerExtraction,
    system_prompt=_UNKNOWN_EXTRACTION_SYSTEM,
    model_settings={"temperature": 0.0}
)


# -----------------------------------------------------------------------------
# ROUTING
# -----------------------------------------------------------------------------

def route(decision: FamilyDecision) -> tuple[Agent, str]:
    if decision.is_supported and decision.family in _AGENTS:
        return _AGENTS[decision.family], _PROMPTS[decision.family]

    custom_name = decision.custom_family_name or "unknown_fastener"
    prompt = (
        f"This fastener is not currently covered by the supported templates. "
        f"The proposed family name is: {custom_name}. "
        f"Extract all visible information and propose a new template."
    )
    return unknown_agent, prompt


# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------

import pydantic_ai.exceptions

def clean_json(raw: str) -> str:
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
    # rimuovi caratteri di controllo eccetto \n \r \t
    raw = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f]', '', raw)
    return raw

_SCHEMA_MAP = {
    FastenerFamily.BOLT: BoltExtraction,
    FastenerFamily.NUT: NutExtraction,
    FastenerFamily.RIVET: RivetExtraction,
    FastenerFamily.WASHER: WasherExtraction,
    FastenerFamily.SCREW: ScrewExtraction,
    FastenerFamily.STUD: StudExtraction,
    FastenerFamily.PIN: PinExtraction,
}

async def process_image(image_path: Path):
    print("  [1/2] Classifying …")
    cls_result = await classifier_agent.run(
        image_message(image_path, "Classify the fastener in this engineering drawing.")
    )
    raw = clean_json(cls_result.output)
    decision = FamilyDecision.model_validate_json(raw)
    print(f"  [1/2] Detected → family={decision.family.value}, supported={decision.is_supported}")

    specialist, prompt = route(decision)
    print("  [2/2] Extracting …")

    schema_class = _SCHEMA_MAP.get(decision.family, UnknownFastenerExtraction)
    schema_json = json.dumps(schema_class.model_json_schema(), indent=2)
    full_prompt = f"{prompt}\n\nRespond ONLY with JSON matching this schema:\n{schema_json}"

    MAX_RETRIES = 3
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            ext_result = await specialist.run(image_message(image_path, full_prompt))
            raw = clean_json(ext_result.output)
            extraction = schema_class.model_validate_json(raw)
            print("  [2/2] Done ✓")
            return decision, extraction
        except Exception as e:
            last_error = e
            print(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            continue

    raise last_error

async def process_directory(
    input_dir: Path,
    output_dir: Path,
) -> list[tuple[Path, FamilyDecision, AnyFastenerExtraction]]:
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
        exists = any(
            f.startswith(png.stem)
            for f in os.listdir(output_dir)
        )
        if exists:
            print("File already processed")
            continue
        
        try:
            decision, extraction = await process_image(png)
            results.append((png, decision, extraction))

            family_name = (
                decision.family.value
                if decision.is_supported
                else normalize_identifier(decision.custom_family_name or "unknown_fastener")
            )

            out_json = output_dir / f"{png.stem}__{family_name}.json"
            out_json.write_text(
                extraction.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8",
            )
            #print(f"  💾 JSON → {out_json.name}")

            if isinstance(extraction, UnknownFastenerExtraction) and extraction.template_proposal:
                code = generate_pydantic_model_code(extraction.template_proposal)
                out_py = template_dir / f"{normalize_identifier(extraction.template_proposal.family_name)}.py"
                out_py.write_text(code, encoding="utf-8")
                print(f"  🧩 Template proposal → {out_py.name}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    return results

async def main() -> None:
    if "model" not in globals():
        raise RuntimeError(
            "Devi definire `model` prima di eseguire questo script. "
            "Configura il tuo provider/model nella sezione MODEL SETUP."
        )

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image.png|directory> [output_dir]")
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
            decision.family.value
            if decision.is_supported
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
            print(f"\nProposed template saved → {out_py}")

class LogTransport(httpx.AsyncBaseTransport):
    def __init__(self, wrapped):
        self._wrapped = wrapped
    
    async def handle_async_request(self, request):
        body = request.content.decode()
        print("\n=== REQUEST BODY ===")
        print(body[:3000])
        print("===================")
        response = await self._wrapped.handle_async_request(request)
        return response



if __name__ == '__main__':
    import asyncio
    
    # Monkey-patch httpx per loggare
    original_async_client = httpx.AsyncClient.__init__
    def patched_init(self, *args, **kwargs):
        original_async_client(self, *args, **kwargs)
        self._transport = LogTransport(self._transport)
    httpx.AsyncClient.__init__ = patched_init
    asyncio.run(main())