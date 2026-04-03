import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


# Setup

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

# -----------------------------
# System prompt
# -----------------------------
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

# -----------------------------
# Profile helpers
# NOTE: Vercel is stateless — profile is passed in from the frontend
# on every request and returned back. The Next.js frontend stores it
# in localStorage or a database. No file system is used here.
# -----------------------------

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

DEFAULT_OUTFIT = {
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
# OpenAI helpers
# -----------------------------

def ask_glowup_bot(
    user_text: str,
    profile: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Send a chat message and return the assistant reply."""
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT + "\n\nSaved profile: " + profile_summary(profile),
    }

    history = messages[-12:] if messages else []

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_message] + history + [{"role": "user", "content": user_text}],
    )
    return response.choices[0].message.content


def extract_profile_updates(
    conversation_text: str,
    current_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Use GPT to pull stable beauty preferences out of conversation text."""
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
            "favorite_colors", "favorite_styles", "best_colors",
            "skin_tone", "undertone", "face_shape",
            "hair_texture", "budget", "notes",
        ],
        "additionalProperties": False,
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only stable beauty/style preferences from the text. "
                        "Do not invent facts. Return empty strings/lists when unknown. "
                        "Respond ONLY with valid JSON matching the schema — no markdown, no extra text."
                    ),
                },
                {"role": "user", "content": conversation_text},
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

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

        return updated

    except Exception:
        return current_profile


def analyze_face_photo(
    image_bytes: bytes,
    mime: str,
    extra_context: str = "",
) -> Dict[str, Any]:
    """Analyze a selfie and return structured beauty guidance."""
    data_url = to_data_url(image_bytes, mime)

    prompt = f"""
Analyze this selfie for beauty guidance. Return ONLY valid JSON — no markdown, no extra text.

Respond with this exact structure:
{{
  "face_shape": "...",
  "skin_tone": "...",
  "undertone": "...",
  "confidence_note": "...",
  "blush_placement": "...",
  "contour_bronzer": "...",
  "lip_shades": ["...", "..."],
  "brow_direction": "...",
  "hairstyle_directions": ["...", "...", "..."],
  "makeup_look": "...",
  "notes": ["..."]
}}

Be cautious. Do not claim certainty. Note in confidence_note if lighting, angle, or image quality reduces confidence.

Extra context from user: {extra_context}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
    )

    return json.loads(response.choices[0].message.content)


def apply_face_analysis_to_profile(
    analysis: Dict[str, Any],
    current_profile: Dict[str, Any],
) -> Dict[str, Any]:
    updated = current_profile.copy()
    for key in ["face_shape", "skin_tone", "undertone"]:
        val = str(analysis.get(key, "")).strip()
        if val:
            updated[key] = val
    updated["notes"] = merge_unique(
        safe_list(updated.get("notes", [])),
        safe_list(analysis.get("notes", [])),
    )
    return updated


def generate_outfit_image(prompt: str) -> str:
    """Generate an outfit image and return base64 string."""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )
    return response.data[0].b64_json


def transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice_question.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text.strip()


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


@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    """
    Send a chat message.
    Payload: { message, messages (history), profile }
    Returns: { reply, profile }
    """
    user_text = payload.get("message", "").strip()
    messages = payload.get("messages", [])
    profile = {**DEFAULT_PROFILE, **payload.get("profile", {})}

    if not user_text:
        raise HTTPException(status_code=400, detail="Missing message.")

    reply = ask_glowup_bot(user_text, profile, messages)

    updated_profile = extract_profile_updates(
        "\n".join(
            [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[-8:]]
            + [f"user: {user_text}", f"assistant: {reply}"]
        ),
        profile,
    )

    return {"reply": reply, "profile": updated_profile}


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
    extra_context: str = Form(default=""),
    profile: str = Form(default="{}"),
):
    """
    Analyze a selfie.
    Form fields: file, extra_context (str), profile (JSON string)
    Returns: { analysis, profile }
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        current_profile = {**DEFAULT_PROFILE, **json.loads(profile)}
    except Exception:
        current_profile = DEFAULT_PROFILE.copy()

    mime = file.content_type or "image/jpeg"
    analysis = analyze_face_photo(image_bytes, mime, extra_context)
    updated_profile = apply_face_analysis_to_profile(analysis, current_profile)

    return {"analysis": analysis, "profile": updated_profile}


@app.post("/generate-image")
async def generate_image_route(payload: Dict[str, Any]):
    """
    Generate an outfit image.
    Payload: { prompt }
    Returns: { image_base64 }
    """
    prompt = payload.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")

    image_b64 = generate_outfit_image(prompt)
    return {"image_base64": image_b64}


@app.post("/outfit-feedback")
async def outfit_feedback_route(payload: Dict[str, Any]):
    """
    Get AI feedback on a drafted outfit.
    Payload: { outfit, profile }
    Returns: { feedback }
    """
    outfit = payload.get("outfit", {})
    profile = {**DEFAULT_PROFILE, **payload.get("profile", {})}

    if not isinstance(outfit, dict):
        raise HTTPException(status_code=400, detail="Invalid outfit payload.")

    prompt = outfit_feedback_prompt(outfit, profile)
    feedback = ask_glowup_bot(prompt, profile)
    return {"feedback": feedback}


@app.post("/extract-profile")
async def extract_profile_route(payload: Dict[str, Any]):
    """
    Extract beauty profile from free text.
    Payload: { text, profile }
    Returns: updated profile dict
    """
    text = payload.get("text", "").strip()
    profile = {**DEFAULT_PROFILE, **payload.get("profile", {})}

    if not text:
        raise HTTPException(status_code=400, detail="Missing text.")

    updated = extract_profile_updates(text, profile)
    return updated