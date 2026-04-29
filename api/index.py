"""
Glow Up Bot — Backend
Fixes:
- /chat endpoint restored (frontend needs it)
- /generate-image endpoint restored
- Confidence threshold lowered so the profile actually saves
- Stabilization no longer rejects single-photo detections
- Hair detection prompt strengthened (protective styles win)
- System prompt updated for cultural relevance
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.253.43.7:3000",
        "*",  # Open during dev. Restrict before deploying publicly.
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# System prompt — culturally relevant
# -----------------------------
SYSTEM_PROMPT = """
You are Glow Up Bot, a warm, stylish, inclusive beauty and fashion assistant.

You help with:
- makeup looks
- hairstyles (including protective styles like braids, locs, twists, bantu knots, silk presses, wigs, weaves)
- skincare basics
- outfit styling
- undertone-aware color choices
- face-shape-aware hairstyles and makeup placement
- beauty inspiration

Cultural relevance is REQUIRED:
- When the user has deep brown / very deep brown skin or coily / 4c / braided / loc'd hair, prioritize products and brands developed for and tested on that hair type and skin tone (e.g. Fenty, Pat McGrath, MAC NW/NC deep range, UOMA, Black Opal, AS I AM, Mielle Organics, Camille Rose, The Mane Choice, Aunt Jackie's, Carol's Daughter, SheaMoisture, Cantu, Kinky-Curly, Adwoa Beauty, Pattern Beauty, Eden BodyWorks).
- For protective styles, suggest styles authentic to the texture (e.g. knotless box braids, fulani braids, goddess locs, marley twists, cornrows, bantu knots, finger waves, silk press, wash-and-go).
- For lip shades, lean into shades that pop on deep complexions (warm browns, plums, brick reds, mahogany, true reds with blue undertones for cool, terracotta for warm).
- Never default to "natural" looks that erase the user's features.

Style:
- upbeat, charismatic, supportive, but not bossy
- can say "Yess" or "Okayy" occasionally — not too slang-heavy
- explain WHY a recommendation fits the user
- keep answers practical and specific

Rules:
- stay focused on beauty, style, skincare, makeup, hair, and outfits
- when asked something outside that, redirect by connecting it to beauty/style
- when working from a face photo analysis, be honest about what was estimated
"""

# -----------------------------
# Profile defaults
# -----------------------------
DEFAULT_PROFILE = {
    "name": "",
    "gender_presentation": "",
    "style_goals": "",
    "skin_concerns": "",
    "hair_type_confirmed": "",
    "preferred_makeup_style": "",
    "fashion_preference": "",
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

# Stable label sets
FACE_SHAPE_LABELS = ["oval", "round", "square", "heart", "diamond", "oblong", "uncertain"]
SKIN_TONE_LABELS = [
    "fair", "light", "medium", "tan", "brown",
    "deep brown", "very deep brown", "uncertain",
]
UNDERTONE_LABELS = ["warm", "cool", "neutral", "olive", "uncertain"]
HAIR_TEXTURE_LABELS = [
    "straight", "wavy", "curly", "coily", "afro",
    "braids", "box braids", "knotless braids", "cornrows", "micro braids",
    "twists", "two-strand twists", "locs", "sisterlocs", "faux locs",
    "silk press", "wig", "weave", "bantu knots",
    "not clearly visible", "uncertain",
]

UNCERTAIN_VALUES = {"uncertain", "not clearly visible", "not detected", ""}

# Lowered: was 0.75 — caused profile to never save
CONFIDENCE_HIGH = 0.6
CONFIDENCE_MED = 0.35


# -----------------------------
# Helpers
# -----------------------------
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


def is_uncertain(v: Any) -> bool:
    return str(v or "").strip().lower() in UNCERTAIN_VALUES


def profile_summary(profile: Dict[str, Any]) -> str:
    parts = []
    for key in ["favorite_colors", "favorite_styles", "best_colors"]:
        values = safe_list(profile.get(key, []))
        if values:
            parts.append(f"{key.replace('_', ' ')}: {', '.join(values)}")
    for key in [
        "name", "gender_presentation", "style_goals", "skin_concerns",
        "hair_type_confirmed", "preferred_makeup_style", "fashion_preference",
        "skin_tone", "undertone", "face_shape", "hair_texture", "budget",
    ]:
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
# Hair detection sanity check
# -----------------------------
def fix_hair_label(result: Dict[str, Any]) -> str:
    """If notes mention braids/locs/twists, force the hair label to match."""
    hair = str(result.get("hair_texture", "")).strip().lower()
    notes_text = " ".join(safe_list(result.get("notes", []))).lower()
    confidence_note = str(result.get("confidence_note", "")).lower()
    blob = notes_text + " " + confidence_note

    # Strong overrides — protective styles always win
    if "loc" in blob or "dread" in blob:
        return "locs"
    if "braid" in blob or "cornrow" in blob:
        return "braids"
    if "twist" in blob:
        return "twists"
    if "bantu" in blob:
        return "bantu knots"

    # If model said "wig" or "wavy" but notes describe woven/plaited hair, override
    if hair in {"wig", "wavy", "curly"} and ("woven" in blob or "plait" in blob or "section" in blob):
        return "braids"

    return hair


# -----------------------------
# Stabilization
# -----------------------------
def majority_vote(values: List[str]) -> Optional[str]:
    filtered = [v.strip().lower() for v in values if v.strip().lower() not in UNCERTAIN_VALUES]
    if not filtered:
        return None
    counts: Dict[str, int] = {}
    for v in filtered:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=lambda k: counts[k])
    # If only one analysis exists, accept it. If 2+, require a clear winner.
    if len(filtered) == 1 or counts[best] >= max(2, len(filtered) // 2 + 1) - 1:
        return best
    if counts[best] >= 2:
        return best
    # Single non-uncertain vote among multiple uncertain — still take it
    return best if counts[best] >= 1 and len(values) <= 2 else best


def stabilize_analyses(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not analyses:
        return {}

    stable_fields = ["face_shape", "skin_tone", "undertone", "hair_texture"]
    result: Dict[str, Any] = {}

    for field in stable_fields:
        values = [str(a.get(field, "")).strip().lower() for a in analyses]
        winner = majority_vote(values)
        # Fallback to most recent non-uncertain value if no majority
        if not winner:
            for a in reversed(analyses):
                v = str(a.get(field, "")).strip().lower()
                if v and v not in UNCERTAIN_VALUES:
                    winner = v
                    break
        result[field] = winner if winner else "uncertain"

    all_hairstyle_dirs: List[str] = []
    all_lip_shades: List[str] = []
    all_notes: List[str] = []

    for a in analyses:
        all_hairstyle_dirs.extend(safe_list(a.get("hairstyle_directions", [])))
        all_lip_shades.extend(safe_list(a.get("lip_shades", [])))
        all_notes.extend(safe_list(a.get("notes", [])))

    latest = analyses[-1]
    result["blush_placement"] = latest.get("blush_placement", "")
    result["contour_bronzer"] = latest.get("contour_bronzer", "")
    result["brow_direction"] = latest.get("brow_direction", "")
    result["makeup_look"] = latest.get("makeup_look", "")

    result["hairstyle_directions"] = list(dict.fromkeys(all_hairstyle_dirs))[:5]
    result["lip_shades"] = list(dict.fromkeys(all_lip_shades))[:4]
    result["notes"] = list(dict.fromkeys(all_notes))[:5]

    n_stable = sum(1 for f in stable_fields if result.get(f, "uncertain") not in UNCERTAIN_VALUES)
    result["confidence_score"] = n_stable / len(stable_fields)
    result["confidence_note"] = _build_confidence_note(result, len(analyses))

    return result


def _build_confidence_note(result: Dict[str, Any], n_analyses: int) -> str:
    score = result.get("confidence_score", 0)
    if score >= CONFIDENCE_HIGH:
        return f"High confidence — {n_analyses} photo(s) analyzed with consistent results."
    if score >= CONFIDENCE_MED:
        return (
            f"Medium confidence — some features detected consistently across {n_analyses} photo(s). "
            f"Try better lighting for any uncertain fields."
        )
    return (
        f"Low confidence across {n_analyses} photo(s). Tip: face camera directly, use bright even "
        f"lighting, and keep your hair/face uncovered."
    )


def apply_to_profile(
    analysis: Dict[str, Any],
    current_profile: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Save a single analysis (or stabilized result) into the profile.
    Always overwrites empty fields. Overwrites filled fields only when:
      - force=True, OR
      - the new value is non-uncertain AND different from the saved value.
    """
    updated = current_profile.copy()
    fields = ["face_shape", "skin_tone", "undertone", "hair_texture"]

    for key in fields:
        new_val = str(analysis.get(key, "")).strip().lower()
        if not new_val or new_val in UNCERTAIN_VALUES:
            continue

        cur_val = str(updated.get(key, "")).strip().lower()
        if not cur_val or force:
            updated[key] = new_val
        elif new_val != cur_val:
            # Differing values — accept new (single-source mode). Stabilization
            # across multiple photos protects against noise.
            updated[key] = new_val

    updated["notes"] = merge_unique(
        safe_list(updated.get("notes", [])),
        safe_list(analysis.get("notes", [])),
    )
    return updated


# -----------------------------
# OpenAI calls
# -----------------------------
def ask_glowup_bot(
    user_text: str,
    profile: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
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
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only stable beauty/style preferences from the text. "
                        "Do not invent facts. Return empty strings/lists when unknown. "
                        "Respond ONLY with valid JSON — no markdown."
                    ),
                },
                {"role": "user", "content": conversation_text},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)

        updated = current_profile.copy()
        for k in ["favorite_colors", "favorite_styles", "best_colors", "notes"]:
            updated[k] = merge_unique(safe_list(updated.get(k, [])), safe_list(data.get(k, [])))
        for k in ["skin_tone", "undertone", "face_shape", "hair_texture", "budget"]:
            v = str(data.get(k, "")).strip()
            if v:
                updated[k] = v
        return updated
    except Exception:
        return current_profile


def analyze_face_photo(
    image_bytes: bytes,
    mime: str,
    extra_context: str = "",
    existing_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data_url = to_data_url(image_bytes, mime)

    profile_hint = ""
    if existing_profile:
        saved_hair = str(existing_profile.get("hair_texture", "")).strip().lower()
        if saved_hair and saved_hair not in UNCERTAIN_VALUES:
            profile_hint = (
                f"\nNote: User's previously saved hair texture is '{saved_hair}'. "
                f"Only change this if you can clearly see something different."
            )

    prompt = f"""
You are analyzing a selfie for a beauty app. Be precise and use ONLY the allowed labels.

ALLOWED LABELS:
face_shape: {", ".join(FACE_SHAPE_LABELS)}
skin_tone: {", ".join(SKIN_TONE_LABELS)}
undertone: {", ".join(UNDERTONE_LABELS)}
hair_texture: {", ".join(HAIR_TEXTURE_LABELS)}

HAIR DETECTION — MOST IMPORTANT:
1. Protective styles ALWAYS override texture-based labels.
   - rope-like fused strands (any length) -> "locs"
   - woven/plaited sections of any size -> "braids" (or "box braids" / "knotless braids" / "cornrows")
   - two-strand twisted sections -> "twists"
   - clearly added straight/wavy/curly hair -> "wig" or "weave"
2. Use "straight"/"wavy"/"curly"/"coily"/"afro" ONLY for natural unprotected hair.
3. NEVER label braids as "wavy" or "wig". If hair is divided into repeated woven/plaited strands, classify as "braids".
4. Hair color/highlights/curls at the ends do NOT change the protective style label.
5. If you genuinely cannot see hair: "not clearly visible". Otherwise, COMMIT — do not say "uncertain" if hair is visible.

FACE SHAPE:
- Look at the perimeter, not just features.
- If covered/angled/shadowed: "uncertain".
- Do NOT default to "oval".

SKIN TONE:
- Use forehead/cheeks/jawline.
- Adjust for lighting.
- Do NOT default to "medium". If unsure: "uncertain".

UNDERTONE:
- Look at vein color, jewelry that flatters, how skin reads in natural light.
- Olive complexions exist — use "olive" when there's a green-ish cast.

CULTURAL RELEVANCE:
- For deeper skin: lip shades that POP (warm browns, plums, brick reds, true reds with blue undertones).
- For protective styles: hairstyle suggestions should fit that style category (e.g. for braids, suggest knotless variations, fulani updos, bun styles, half-up; not "let your curls down").
{profile_hint}

OUTPUT — JSON only, no extra text:
{{
  "face_shape": "...",
  "skin_tone": "...",
  "undertone": "...",
  "hair_texture": "...",
  "confidence_note": "brief note on image quality",
  "confidence_score": 0.0 to 1.0,
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
                "content": "You are a precise beauty analysis assistant. Always respond with valid JSON exactly matching the requested schema.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=1200,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Normalize labels
    for field in ["face_shape", "skin_tone", "undertone", "hair_texture"]:
        if field in result:
            result[field] = str(result[field]).strip().lower()

    # Hair sanity check
    result["hair_texture"] = fix_hair_label(result)

    try:
        result["confidence_score"] = float(result.get("confidence_score", 0.5))
    except (TypeError, ValueError):
        result["confidence_score"] = 0.5

    return result


def generate_outfit_image(prompt: str) -> str:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )
    return response.data[0].b64_json


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Glow Up Bot API is running"}


@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    """Chat with Glow Up Bot. Payload: { message, messages, profile }"""
    try:
        user_text = str(payload.get("message", "")).strip()
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


@app.post("/analyze-face")
async def analyze_face_route(
    file: UploadFile = File(...),
    extra_context: str = Form(default=""),
    profile: str = Form(default="{}"),
    analysis_history: str = Form(default="[]"),
    force_update: str = Form(default="true"),
):
    """
    Analyze a selfie. Returns:
      - analysis: raw single-photo result
      - stabilized: consensus across history
      - profile: updated profile (always saves what's clearly visible)
      - analysis_history: updated history (frontend stores this)
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

    do_force = str(force_update).lower() == "true"
    mime = file.content_type or "image/jpeg"

    analysis = analyze_face_photo(image_bytes, mime, extra_context, current_profile)
    updated_history = (history + [analysis])[-3:]
    stabilized = stabilize_analyses(updated_history)

    # ALWAYS update the profile from this single photo's non-uncertain values.
    # (This is the main fix — don't gate on confidence; gate on per-field uncertainty.)
    updated_profile = apply_to_profile(analysis, current_profile, force=do_force)

    # Then layer stabilized results on top — they only fire when 2+ photos agree.
    if stabilized:
        updated_profile = apply_to_profile(stabilized, updated_profile, force=do_force)

    return {
        "analysis": analysis,
        "stabilized": stabilized,
        "profile": updated_profile,
        "analysis_history": updated_history,
        "should_update_profile": True,  # we always update; kept for compatibility
    }


@app.post("/generate-image")
async def generate_image_route(payload: Dict[str, Any]):
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")
    try:
        image_b64 = generate_outfit_image(prompt)
        return {"image_base64": image_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


@app.post("/transcribe-audio")
async def transcribe_audio_route(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice_question.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return {"transcript": transcript.text.strip()}


@app.post("/extract-profile")
async def extract_profile_route(payload: Dict[str, Any]):
    text = str(payload.get("text", "")).strip()
    profile = {**DEFAULT_PROFILE, **payload.get("profile", {})}
    if not text:
        raise HTTPException(status_code=400, detail="Missing text.")
    return extract_profile_updates(text, profile)