import os, re, json
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")

MISTRAL_API_KEY   = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_URL       = "https://api.mistral.ai/v1/chat/completions"

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_URL     = "https://api.elevenlabs.io/v1/text-to-speech"

# Map legacy voice names → ElevenLabs voice IDs (multilingual v2, Japanese-capable)
# Override via env vars if needed
VOICE_IDS = {
    "jf_alpha":      os.environ.get("VOICE_JF_ALPHA",      "EXAVITQu4vr4xnSDxMaL"),  # Sarah (F)
    "jf_gongitsune": os.environ.get("VOICE_JF_GONGITSUNE", "9BWtsMINqrJLrRacOk9x"),  # Aria  (F)
    "jm_kumo":       os.environ.get("VOICE_JM_KUMO",       "IKne3meq5aSn9XLyUdCD"),  # Charlie (M)
}
DEFAULT_VOICE = "jf_alpha"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are playing a role in a Japanese conversation practice scenario with a learner.

## Your role
You will be given a context and a specific character to play. Stay in character at ALL times.
Use natural, authentic Japanese that a real Japanese person in that role would use — not textbook Japanese.

Examples of natural Japanese by role:
- Shop staff: いらっしゃいませ！何になさいますか？/ ～円になります。/ 少々お待ちください。
- Doctor: どうされましたか？/ いつ頃から？/ お大事に。
- Friend: ねえ、聞いた？/ マジで！？/ それ、やばくない？
- Interviewer: 本日はよろしくお願いします。/ 志望動機を教えていただけますか？

## Language rules
- **Speak Japanese by default** — stay immersed in the role.
- **Switch to {ui_lang_name}** ONLY when the learner:
  1. Speaks {ui_lang_name} to you
  2. Explicitly asks for explanation in their language
- After a {ui_lang_name} explanation, return to Japanese and the role-play.
- If the learner makes a mistake, gently correct them IN CHARACTER (e.g. repeat the correct phrase naturally).

## Response format
Always reply as JSON — no text outside JSON:
{{
  "reply_text": "What to DISPLAY. Japanese with furigana for hard kanji e.g. 注文（ちゅうもん）. Or {ui_lang_name} if explaining.",
  "tts_text": "Japanese to READ ALOUD. Must be natural spoken Japanese. Never empty.",
  "romaji": "Romaji of tts_text",
  "translation": "{ui_lang_name} meaning of tts_text"
}}

## Level adjustments
{level_instruction}

## Keep the conversation alive — ALWAYS
NEVER end a response without driving the conversation forward. Always append a question or next action. Examples:
- After seating guest → 「お飲み物はいかがでしょうか？」or「メニューをどうぞ。ご注文がお決まりになりましたら、お呼びください。」
- After taking order → 「お飲み物はいかがですか？」or「以上でよろしいでしょうか？」
- After confirming order → 「お持ち帰りですか、それともこちらでお召し上がりですか？」
- After payment → 「またのご来店をお待ちしております！」
If you just gave info or did an action, ALWAYS follow with a question or invitation to continue.
One response = action/statement + follow-up question. Never just a statement alone.

## Keigo (politeness) rules — STRICTLY follow
- Service staff (店員、受付、医者、面接官 etc.) → always use 敬語: ～でございます、～になさいますか、少々お待ちください、ありがとうございます
- Friends / classmates → casual: ～だよ、～じゃない？、ね、よ
- Never mix keigo and casual in the same role
- Real examples for staff:
  ✅ 何になさいますか？/ ご注文はお決まりでしょうか？/ 少々お待ちくださいませ
  ❌ 何にしますか？/ 何を買いますか？/ 待ってください

## Important
- Keep responses SHORT and natural (1-3 sentences) like real conversation
- Do NOT over-explain or add parenthetical romaji in tts_text
- tts_text must be PURE Japanese with no romaji, brackets, or annotations
"""

LEVEL_INSTRUCTIONS = {
    "beginner":     "Learner is a complete beginner (JLPT N5). Use simple Japanese but NEVER sacrifice politeness for simplicity — staff/service roles must still use proper keigo (いらっしゃいませ、何になさいますか、～でございます). Add furigana for all kanji in reply_text.",
    "elementary":   "Learner is elementary (JLPT N5-N4). Use simple natural Japanese. Service roles use proper keigo. Add furigana for harder kanji.",
    "intermediate": "Learner is intermediate (JLPT N3-N4). Use natural Japanese appropriate to the role. Service roles use keigo, friends use casual speech. Furigana only for N2+ kanji.",
    "advanced":     "Learner is advanced (JLPT N1-N2). Use fully natural Japanese. Keigo for formal roles, casual/slang for friend roles. No furigana needed.",
}

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "MISTRAL_API_KEY not set"}), 500

    data    = request.json
    history = data.get("history", [])
    level   = data.get("level", "beginner")
    ui_lang = data.get("ui_lang", "vi")
    lang_names = {"vi": "Vietnamese", "en": "English", "ja": "Japanese"}
    ui_lang_name = lang_names.get(ui_lang, "Vietnamese")

    system   = SYSTEM_PROMPT.format(
        level_instruction=LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["beginner"]),
        ui_lang_name=ui_lang_name
    )
    messages = [{"role": "system", "content": system}] + history

    resp = requests.post(
        MISTRAL_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={"model": "mistral-large-latest", "messages": messages, "max_tokens": 600, "temperature": 0.7}
    )
    if not resp.ok:
        return jsonify({"error": resp.text}), resp.status_code

    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = {"reply_text": raw, "tts_text": "", "romaji": "", "translation": ""}

    return jsonify(parsed)


@app.route("/tts", methods=["POST"])
def tts():
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "ELEVENLABS_API_KEY not set"}), 500

    data       = request.json
    text       = data.get("text", "").strip()
    voice_name = data.get("voice", DEFAULT_VOICE)

    if not text:
        return jsonify({"error": "no text"}), 400

    voice_id = VOICE_IDS.get(voice_name, VOICE_IDS[DEFAULT_VOICE])

    resp = requests.post(
        f"{ELEVENLABS_URL}/{voice_id}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
    )
    if not resp.ok:
        return jsonify({"error": resp.text}), resp.status_code

    return Response(resp.content, mimetype="audio/mpeg")


@app.route("/analyze", methods=["POST"])
def analyze():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "MISTRAL_API_KEY not set"}), 500

    data  = request.json
    text  = data.get("text", "").strip()
    level = data.get("level", "beginner")
    ui_lang = data.get("ui_lang", "vi")
    lang_names = {"vi": "Vietnamese", "en": "English", "ja": "Japanese"}
    ui_lang_name = lang_names.get(ui_lang, "Vietnamese")

    if not text:
        return jsonify({"error": "no text"}), 400

    prompt = f"""Analyze this message from a Japanese learner (level: {level}): "{text}"

Reply ONLY as JSON, no extra text:
{{
  "is_japanese": true/false,  // true if the message contains Japanese attempt
  "is_correct": true/false,   // grammatically and naturally correct?
  "verdict": "one short sentence in {ui_lang_name} summarizing correctness",
  "correction": "corrected Japanese sentence if wrong, else null",
  "tip": "one short {ui_lang_name} tip about grammar/vocabulary/politeness if relevant, else null",
  "japanese": "if not Japanese input: how to say it in Japanese",
  "romaji": "romaji of japanese field if provided"
}}

Be concise. verdict max 10 words. tip max 15 words."""

    resp = requests.post(
        MISTRAL_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0.3}
    )
    if not resp.ok:
        return jsonify({"error": resp.text}), resp.status_code

    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        return jsonify(extract_json(raw))
    except Exception:
        return jsonify({"error": "parse error"}), 500


@app.route("/translate", methods=["POST"])
def translate_word():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "MISTRAL_API_KEY not set"}), 500

    data = request.json
    word = data.get("word", "").strip()
    ui_lang = data.get("ui_lang", "vi")
    lang_names = {"vi": "Vietnamese", "en": "English", "ja": "Japanese"}
    ui_lang_name = lang_names.get(ui_lang, "Vietnamese")
    if not word:
        return jsonify({"error": "no word"}), 400

    prompt = f"""Translate this Japanese word or phrase to {ui_lang_name}: "{word}"
Reply ONLY as JSON, no extra text:
{{"translation": "concise {ui_lang_name} meaning (max 8 words)", "romaji": "romaji reading"}}
Keep translation concise (max 8 words)."""

    resp = requests.post(
        MISTRAL_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100, "temperature": 0.2}
    )
    if not resp.ok:
        return jsonify({"error": resp.text}), resp.status_code

    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        return jsonify(extract_json(raw))
    except Exception:
        return jsonify({"error": "parse error"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "tts": "elevenlabs"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
