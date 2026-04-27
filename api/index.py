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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://10.253.43.7:3000"],
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
# in localStorage. No file system is used here.
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

# Stable labels the model must use
FACE_SHAPE_LABELS = ["oval", "round", "square", "heart", "diamond", "oblong", "uncertain"]
SKIN_TONE_LABELS = ["fair", "light", "medium", "tan", "brown", "deep brown", "very deep brown", "uncertain"]
UNDERTONE_LABELS = ["warm", "cool", "neutral", "olive", "uncertain"]

# Hair texture labels — protective styles are first-class labels
HAIR_TEXTURE_LABELS = [
    "straight", "wavy", "curly", "coily", "afro",
    "braids", "box braids", "knotless braids", "cornrows", "micro braids",
    "twists", "two-strand twists", "locs", "sisterlocs", "faux locs",
    "silk press", "wig", "weave", "bantu knots",
    "not clearly visible", "uncertain"
]

UNCERTAIN_VALUES = {"uncertain", "not clearly visible", "not detected", ""}


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
# Stabilization helpers
# -----------------------------

def majority_vote(values: List[str]) -> Optional[str]:
    """Return the most common non-uncertain value from a list, or None."""
    filtered = [v.strip().lower() for v in values if v.strip().lower() not in UNCERTAIN_VALUES]
    if not filtered:
        return None
    counts: Dict[str, int] = {}
    for v in filtered:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=lambda k: counts[k])
    # Require at least 2 votes out of 3 to be considered stable
    if counts[best] >= 2:
        return best
    return None


def stabilize_analyses(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given up to 3 recent analysis results, return a stabilized consensus.
    Only fields where at least 2/3 analyses agree get a non-uncertain value.
    """
    if not analyses:
        return {}

    stable_fields = ["face_shape", "skin_tone", "undertone", "hair_texture"]
    result: Dict[str, Any] = {}

    for field in stable_fields:
        values = [str(a.get(field, "")).strip().lower() for a in analyses]
        winner = majority_vote(values)
        result[field] = winner if winner else "uncertain"

    # For list fields, merge across all analyses
    all_hairstyle_dirs: List[str] = []
    all_lip_shades: List[str] = []
    all_notes: List[str] = []

    for a in analyses:
        all_hairstyle_dirs.extend(safe_list(a.get("hairstyle_directions", [])))
        all_lip_shades.extend(safe_list(a.get("lip_shades", [])))
        all_notes.extend(safe_list(a.get("notes", [])))

    # Use the most recent analysis for cosmetic suggestions
    latest = analyses[-1]
    result["blush_placement"] = latest.get("blush_placement", "")
    result["contour_bronzer"] = latest.get("contour_bronzer", "")
    result["brow_direction"] = latest.get("brow_direction", "")
    result["makeup_look"] = latest.get("makeup_look", "")

    # De-duplicate list fields
    result["hairstyle_directions"] = list(dict.fromkeys(all_hairstyle_dirs))[:5]
    result["lip_shades"] = list(dict.fromkeys(all_lip_shades))[:4]
    result["notes"] = list(dict.fromkeys(all_notes))[:5]

    # Confidence score: fraction of stable fields that are not uncertain
    n_stable = sum(1 for f in stable_fields if result.get(f, "uncertain") != "uncertain")
    result["confidence_score"] = n_stable / len(stable_fields)
    result["confidence_note"] = _build_confidence_note(result, len(analyses))

    return result


def _build_confidence_note(result: Dict[str, Any], n_analyses: int) -> str:
    score = result.get("confidence_score", 0)
    if score >= 0.75:
        return f"High confidence — {n_analyses} photo(s) analyzed with consistent results."
    elif score >= 0.5:
        return f"Medium confidence — some features detected consistently across {n_analyses} photo(s). Try better lighting for remaining fields."
    else:
        return f"Low confidence across {n_analyses} photo(s). For best results: face the camera directly, use bright even lighting, and keep your face uncovered."


def apply_stabilized_to_profile(
    stabilized: Dict[str, Any],
    current_profile: Dict[str, Any],
    force_update: bool = False,
) -> Dict[str, Any]:
    """
    Update profile with stabilized results.
    Only overwrite a field if:
      - The new value is not uncertain, AND
      - The profile field is currently empty OR force_update is True
    This prevents a single low-quality photo from overwriting good saved data.
    """
    updated = current_profile.copy()
    stable_fields = ["face_shape", "skin_tone", "undertone", "hair_texture"]

    for key in stable_fields:
        new_val = str(stabilized.get(key, "")).strip().lower()
        current_val = str(updated.get(key, "")).strip().lower()

        if new_val in UNCERTAIN_VALUES:
            continue  # Never overwrite with uncertain

        if not current_val or force_update:
            updated[key] = new_val
        elif new_val != current_val:
            # Only update if confidence is high enough to override
            if stabilized.get("confidence_score", 0) >= 0.75:
                updated[key] = new_val

    updated["notes"] = merge_unique(
        safe_list(updated.get("notes", [])),
        safe_list(stabilized.get("notes", [])),
    )

    return updated


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

    history = messages[-12:] if messages else [{"role": "user", "content": user_text}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_message] + history,
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
    existing_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze a selfie and return structured beauty guidance.
    Improved prompting for protective styles and consistency.
    """
    data_url = to_data_url(image_bytes, mime)

    profile_hint = ""
    if existing_profile:
        saved_hair = existing_profile.get("hair_texture", "")
        if saved_hair and saved_hair.lower() not in UNCERTAIN_VALUES:
            profile_hint = f"\nNote: The user's previously saved hair texture is '{saved_hair}'. Only change this if you can clearly see something different."

    hair_labels = ", ".join(HAIR_TEXTURE_LABELS)
    face_labels = ", ".join(FACE_SHAPE_LABELS)
    skin_labels = ", ".join(SKIN_TONE_LABELS)
    undertone_labels = ", ".join(UNDERTONE_LABELS)

    prompt = f"""
You are analyzing a selfie for a beauty app. Be precise, honest, and use ONLY the allowed labels.

=== ALLOWED LABELS ===
face_shape: {face_labels}
skin_tone: {skin_labels}
undertone: {undertone_labels}
hair_texture: {hair_labels}

=== HAIR TEXTURE DETECTION — CRITICAL ===
This is the most important field. Read carefully:

1. LOOK FIRST at the hair before classifying. Ask: Are there individual rope-like strands that could be locs? Are strands woven/plaited into braids? Are there defined coil or curl patterns?
2. PROTECTIVE STYLES override texture-based labels:
   - If you see LOCS (rope-like fused strands, any length): return "locs"
   - If you see BRAIDS of any kind (box braids, cornrows, knotless, micro): return "braids"  
   - If you see TWISTS (two-strand twisted sections): return "twists"
   - If you see a WIG or WEAVE that is clearly added hair: return "wig" or "weave"
3. Only use "straight", "wavy", "curly", "coily", "afro" for NATURAL unprotected hair
4. If you genuinely cannot see the hair (covered, off-frame, very blurry): return "not clearly visible"
5. NEVER return "uncertain" for hair if you can see it at all — commit to your best label

=== FACE ANALYSIS ===
- Look for the overall face perimeter shape, not just features
- If the face is partially covered, angled, or shadowed: return "uncertain"
- Do NOT default to "oval" — only return oval if it's clearly oval

=== SKIN TONE ===
- Use visible skin on the face (forehead, cheeks, jawline)
- Adjust for lighting: bright lighting can wash out skin, dim lighting can deepen it
- Do NOT default to "medium" when uncertain — return "uncertain"
- Choose from: fair, light, medium, tan, brown, deep brown, very deep brown, uncertain

=== CONSISTENCY RULE ===
Only report what you can clearly see. If uncertain, say "uncertain". Do not guess to fill in a field.
{profile_hint}

=== OUTPUT FORMAT ===
Respond with ONLY this JSON object, no extra text:
{{
  "face_shape": "one label from the allowed list",
  "skin_tone": "one label from the allowed list",
  "undertone": "one label from the allowed list",
  "hair_texture": "one label from the allowed list",
  "confidence_note": "brief note on image quality and what affected confidence",
  "confidence_score": 0.0 to 1.0 as a number,
  "blush_placement": "short description",
  "contour_bronzer": "short description",
  "lip_shades": ["shade 1", "shade 2"],
  "brow_direction": "short description",
  "hairstyle_directions": ["direction 1", "direction 2", "direction 3"],
  "makeup_look": "short description",
  "notes": ["any important observation"]
}}

Extra context from user: {extra_context}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a precise beauty analysis assistant. Always respond with a valid JSON object exactly matching the requested schema."
            },
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
        max_tokens=1200,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Normalize all label fields to lowercase
    for field in ["face_shape", "skin_tone", "undertone", "hair_texture"]:
        if field in result:
            result[field] = str(result[field]).strip().lower()

    # Ensure confidence_score is a float
    try:
        result["confidence_score"] = float(result.get("confidence_score", 0.5))
    except (TypeError, ValueError):
        result["confidence_score"] = 0.5

    return result


def apply_face_analysis_to_profile(
    analysis: Dict[str, Any],
    current_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Legacy single-analysis profile update.
    Only updates fields that are non-uncertain.
    """
    updated = current_profile.copy()
    stable_keys = ["face_shape", "skin_tone", "undertone", "hair_texture"]

    for key in stable_keys:
        val = str(analysis.get(key, "")).strip().lower()
        if val and val not in UNCERTAIN_VALUES:
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
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        print("CHAT ROUTE ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")


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
    analysis_history: str = Form(default="[]"),
    force_update: str = Form(default="false"),
):
    """
    Analyze a selfie with consistency/stabilization logic.

    Form fields:
      file            — image file
      extra_context   — string of extra context from the user
      profile         — JSON string of current profile
      analysis_history — JSON string of last 0-2 previous raw analyses
      force_update    — "true" to force profile update regardless of confidence

    Returns: { analysis, stabilized, profile, should_update_profile }
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        current_profile = {**DEFAULT_PROFILE, **json.loads(profile)}
    except Exception:
        current_profile = DEFAULT_PROFILE.copy()

    try:
        history: List[Dict[str, Any]] = json.loads(analysis_history)
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []

    do_force_update = str(force_update).lower() == "true"

    mime = file.content_type or "image/jpeg"

    # Run the analysis
    analysis = analyze_face_photo(image_bytes, mime, extra_context, current_profile)

    # Add this analysis to history and keep last 3
    updated_history = (history + [analysis])[-3:]

    # Stabilize across all analyses in history
    stabilized = stabilize_analyses(updated_history)

    # Decide whether to update profile
    # Update if: high confidence OR force_update OR profile fields are still empty
    profile_fields_empty = all(
        not str(current_profile.get(f, "")).strip()
        for f in ["face_shape", "skin_tone", "undertone", "hair_texture"]
    )

    should_update = (
        do_force_update
        or profile_fields_empty
        or stabilized.get("confidence_score", 0) >= 0.6
    )

    if should_update:
        updated_profile = apply_stabilized_to_profile(stabilized, current_profile, force_update=do_force_update)
    else:
        updated_profile = current_profile

    return {
        "analysis": analysis,           # Raw single-photo result
        "stabilized": stabilized,       # Consensus across history
        "profile": updated_profile,     # Updated profile (or unchanged)
        "analysis_history": updated_history,  # Updated history for frontend to store
        "should_update_profile": should_update,
    }


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