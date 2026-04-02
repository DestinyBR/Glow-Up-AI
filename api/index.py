import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

client = OpenAI(api_key=api_key)

app = FastAPI(title="Glow Up Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PROFILE_PATH = DATA_DIR / "profile.json"
OUTFITS_PATH = DATA_DIR / "saved_outfits.json"

SYSTEM_PROMPT = """
You are Glow Up Bot, a warm, stylish, inclusive beauty and fashion assistant.

You help with:
- makeup looks
- hairstyles
- skincare basics
- outfit styling
- undertone-aware color choices
- face-shape-aware hairstyles and suggested makeup placement
- beauty inspiration images

Style:
- upbeat but clear
- charismatic and confident, but not bossy
- supportive and specific
- can say things like "Yess" or "Okayy" occasionally
- not too slang-heavy

Rules:
- stay focused on beauty, style, skincare, makeup, hair, and outfits
- when asked a question outside your expertise, redirect the user back to your beauty and fashion superpowers by connecting their question to beauty and style
- when analyzing a face photo, be careful and say when something is only an estimate
- explain WHY a recommendation fits the user
- keep answers practical
"""

DEFAULT_PROFILE = {
    "name": "",
    "favorite_colors": [],
    "favorite_styles": [],
    "best_colors": [],
    "skin_tone": "",
    "undertone": "",
    "face_shape": "",
    "hair_texture": "",
    "budget": "",
    "notes": [],
}

DEFAULT_OUTFIT_GAME = {
    "occasion": "",
    "vibe": "",
    "top": "",
    "bottom": "",
    "dress": "",
    "outerwear": "",
    "shoes": "",
    "bag": "",
    "accessories": "",
    "colors": "",
    "notes": "",
}


# -----------------------------
# Persistence helpers
# -----------------------------
def load_json(path: Path, fallback: Any) -> Any:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return fallback
    return fallback


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def merge_unique(old_items: List[str], new_items: List[str]) -> List[str]:
    seen = {item.lower(): item for item in old_items}
    for item in new_items:
        clean = str(item).strip()
        if clean and clean.lower() not in seen:
            seen[clean.lower()] = clean
    return list(seen.values())


def profile_summary(profile: Dict[str, Any]) -> str:
    parts = []
    for key in ["favorite_colors", "favorite_styles", "best_colors"]:
        values = safe_list(profile.get(key, []))
        if values:
            parts.append(f"{key.replace('_', ' ')}: {', '.join(values)}")
    for key in ["skin_tone", "undertone", "face_shape", "hair_texture", "budget"]:
        val = str(profile.get(key, "")).strip()
        if val:
            parts.append(f"{key.replace('_', ' ')}: {val}")
    note_values = safe_list(profile.get("notes", []))
    if note_values:
        parts.append(f"notes: {', '.join(note_values[:5])}")
    return " | ".join(parts) if parts else "No saved profile yet."


def to_data_url(file_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"


# -----------------------------
# Data access
# -----------------------------
def get_profile() -> Dict[str, Any]:
    saved_profile = load_json(PROFILE_PATH, DEFAULT_PROFILE.copy())
    merged = DEFAULT_PROFILE.copy()
    merged.update(saved_profile)
    return merged


def set_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    merged = DEFAULT_PROFILE.copy()
    merged.update(profile)
    save_json(PROFILE_PATH, merged)
    return merged


def get_saved_outfits() -> List[Dict[str, Any]]:
    return load_json(OUTFITS_PATH, [])


def set_saved_outfits(outfits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    save_json(OUTFITS_PATH, outfits)
    return outfits


# -----------------------------
# OpenAI helpers
# -----------------------------
def extract_profile_updates(conversation_text: str, current_profile: Dict[str, Any]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "favorite_colors": {"type": "array", "items": {"type": "string"}},
            "favorite_styles": {"type": "array", "items": {"type": "string"}},
            "best_colors": {"type": "array", "items": {"type": "string"}},
            "skin_tone": {"type": "string"},
            "undertone": {"type": "string"},
            "face_shape": {"type": "string"},
            "hair_texture": {"type": "string"},
            "budget": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "favorite_colors",
            "favorite_styles",
            "best_colors",
            "skin_tone",
            "undertone",
            "face_shape",
            "hair_texture",
            "budget",
            "notes",
        ],
        "additionalProperties": False,
    }

    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": (
                        "Extract only stable beauty/style preferences from the text. "
                        "Do not invent facts. Return empty strings/lists when unknown."
                    ),
                },
                {"role": "user", "content": conversation_text},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "profile_update",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        data = json.loads(response.output_text)

        updated = current_profile.copy()
        updated["favorite_colors"] = merge_unique(
            safe_list(updated.get("favorite_colors", [])),
            safe_list(data.get("favorite_colors", [])),
        )
        updated["favorite_styles"] = merge_unique(
            safe_list(updated.get("favorite_styles", [])),
            safe_list(data.get("favorite_styles", [])),
        )
        updated["best_colors"] = merge_unique(
            safe_list(updated.get("best_colors", [])),
            safe_list(data.get("best_colors", [])),
        )
        updated["notes"] = merge_unique(
            safe_list(updated.get("notes", [])),
            safe_list(data.get("notes", [])),
        )

        for key in ["skin_tone", "undertone", "face_shape", "hair_texture", "budget"]:
            val = str(data.get(key, "")).strip()
            if val:
                updated[key] = val

        save_json(PROFILE_PATH, updated)
        return updated
    except Exception:
        return current_profile


def build_model_messages(user_text: str, profile: Dict[str, Any], messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    prompt_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nSaved profile: " + profile_summary(profile),
        }
    ]
    if messages:
        prompt_messages.extend(messages[-12:])
    prompt_messages.append({"role": "user", "content": user_text})
    return prompt_messages


def ask_glowup_bot(user_text: str, profile: Dict[str, Any], messages: Optional[List[Dict[str, Any]]] = None) -> str:
    response = client.responses.create(
        model="gpt-5",
        input=build_model_messages(user_text, profile, messages),
        truncation="auto",
    )
    return response.output_text


def transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice_question.wav"
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
    )
    return getattr(transcript, "text", "").strip()


def analyze_face_photo_structured(image_bytes: bytes, mime: str, extra_context: str = "") -> Dict[str, Any]:
    data_url = to_data_url(image_bytes, mime)

    schema = {
        "type": "object",
        "properties": {
            "face_shape": {"type": "string"},
            "skin_tone": {"type": "string"},
            "undertone": {"type": "string"},
            "confidence_note": {"type": "string"},
            "blush_placement": {"type": "string"},
            "contour_bronzer": {"type": "string"},
            "lip_shades": {"type": "array", "items": {"type": "string"}},
            "brow_direction": {"type": "string"},
            "hairstyle_directions": {"type": "array", "items": {"type": "string"}},
            "makeup_look": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "face_shape",
            "skin_tone",
            "undertone",
            "confidence_note",
            "blush_placement",
            "contour_bronzer",
            "lip_shades",
            "brow_direction",
            "hairstyle_directions",
            "makeup_look",
            "notes",
        ],
        "additionalProperties": False,
    }

    prompt = f"""
Analyze this selfie for beauty guidance.

Return your best estimate for:
- likely face shape
- visible skin tone
- likely undertone

Be cautious.
Do not claim certainty.
If lighting, angle, makeup, or image quality reduces confidence, say so in confidence_note.

Also provide:
- flattering blush placement
- contour/bronzer strategy
- lip shades
- brow direction
- 3 hairstyle directions
- 1 makeup look that would flatter the face

Extra context from the user:
{extra_context}
"""

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "face_analysis",
                "schema": schema,
                "strict": True,
            }
        },
        truncation="auto",
    )

    return json.loads(response.output_text)


def apply_face_analysis_to_profile(analysis_data: Dict[str, Any], current_profile: Dict[str, Any]) -> Dict[str, Any]:
    updated = current_profile.copy()

    face_shape = str(analysis_data.get("face_shape", "")).strip()
    skin_tone = str(analysis_data.get("skin_tone", "")).strip()
    undertone = str(analysis_data.get("undertone", "")).strip()
    notes = safe_list(analysis_data.get("notes", []))

    if face_shape:
        updated["face_shape"] = face_shape
    if skin_tone:
        updated["skin_tone"] = skin_tone
    if undertone:
        updated["undertone"] = undertone

    updated["notes"] = merge_unique(
        safe_list(updated.get("notes", [])),
        notes
    )

    save_json(PROFILE_PATH, updated)
    return updated


def generate_outfit_image(prompt: str) -> Optional[str]:
    response = client.responses.create(
        model="gpt-5",
        input=prompt,
        tools=[{"type": "image_generation"}],
        tool_choice={"type": "image_generation"},
    )

    if hasattr(response, "output"):
        for item in response.output:
            item_type = getattr(item, "type", "")
            if item_type == "image_generation_call":
                result = getattr(item, "result", None)
                if isinstance(result, str):
                    return result

                if isinstance(result, list):
                    for piece in result:
                        b64_data = getattr(piece, "b64_json", None) or getattr(piece, "image_base64", None)
                        if b64_data:
                            return b64_data

            if hasattr(item, "result"):
                result = getattr(item, "result")
                if isinstance(result, str):
                    return result

    return None


def outfit_feedback_prompt(game: Dict[str, str], profile: Dict[str, Any]) -> str:
    return f"""
We are building an outfit together.

Saved user profile:
{profile_summary(profile)}

Current outfit draft:
- Occasion: {game.get('occasion', '')}
- Vibe: {game.get('vibe', '')}
- Top: {game.get('top', '')}
- Bottom: {game.get('bottom', '')}
- Dress: {game.get('dress', '')}
- Outerwear: {game.get('outerwear', '')}
- Shoes: {game.get('shoes', '')}
- Bag: {game.get('bag', '')}
- Accessories: {game.get('accessories', '')}
- Colors: {game.get('colors', '')}
- Notes: {game.get('notes', '')}

Please:
1. Say what works
2. Point out what clashes or feels incomplete
3. Suggest improvements
4. End with a short section called "Final Outfit Check"
"""


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Glow Up Bot API is running"}


@app.get("/profile")
def read_profile():
    return get_profile()


@app.post("/profile")
async def update_profile(payload: Dict[str, Any]):
    current = get_profile()
    current.update(payload)
    return set_profile(current)


@app.post("/profile/extract")
async def extract_profile(payload: Dict[str, Any]):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text.")
    updated = extract_profile_updates(text, get_profile())
    return updated


@app.get("/outfits")
def read_outfits():
    return get_saved_outfits()


@app.post("/outfits")
async def save_outfit(payload: Dict[str, Any]):
    outfits = get_saved_outfits()
    outfits.insert(0, payload)
    set_saved_outfits(outfits)
    return {"success": True, "outfits": outfits}


@app.delete("/outfits/{index}")
def delete_outfit(index: int):
    outfits = get_saved_outfits()
    if index < 0 or index >= len(outfits):
        raise HTTPException(status_code=404, detail="Outfit not found.")
    deleted = outfits.pop(index)
    set_saved_outfits(outfits)
    return {"success": True, "deleted": deleted, "outfits": outfits}


@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    user_text = payload.get("message", "").strip()
    messages = payload.get("messages", [])
    if not user_text:
        raise HTTPException(status_code=400, detail="Missing message.")

    profile = get_profile()
    reply = ask_glowup_bot(user_text, profile, messages)
    updated_profile = extract_profile_updates(
        "\n".join(
            [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[-8:]]
            + [f"user: {user_text}", f"assistant: {reply}"]
        ),
        profile,
    )

    return {
        "reply": reply,
        "profile": updated_profile,
    }


@app.post("/transcribe-audio")
async def transcribe_audio_route(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    text = transcribe_audio(audio_bytes)
    return {"transcript": text}


@app.post("/analyze-face")
async def analyze_face_route(
    file: UploadFile = File(...),
    extra_context: str = Form(default="")
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    mime = file.content_type or "image/jpeg"
    analysis = analyze_face_photo_structured(image_bytes, mime, extra_context)
    updated_profile = apply_face_analysis_to_profile(analysis, get_profile())

    return {
        "analysis": analysis,
        "profile": updated_profile,
    }


@app.post("/generate-image")
async def generate_image_route(payload: Dict[str, Any]):
    prompt = payload.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")

    image_b64 = generate_outfit_image(prompt)
    if not image_b64:
        raise HTTPException(status_code=500, detail="No image returned.")

    return {
        "image_base64": image_b64
    }


@app.post("/outfit-feedback")
async def outfit_feedback_route(payload: Dict[str, Any]):
    outfit = payload.get("outfit", {})
    if not isinstance(outfit, dict):
        raise HTTPException(status_code=400, detail="Invalid outfit payload.")

    prompt = outfit_feedback_prompt(outfit, get_profile())
    reply = ask_glowup_bot(prompt, get_profile())
    return {"feedback": reply}