from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import json
import re

app = Flask(__name__)
CORS(app, origins="*")

MISTRAL_API_KEY    = os.environ.get("MISTRAL_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")  # Japanese voice ID
MISTRAL_URL        = "https://api.mistral.ai/v1/chat/completions"
ELEVENLABS_URL     = "https://api.elevenlabs.io/v1/text-to-speech"

SYSTEM_PROMPT = """You are Sensei Yuki, a warm and encouraging Japanese language teacher for Vietnamese learners.

## Core behavior
- **Speak Japanese by default** in every response — this is an immersive lesson.
- **Switch to Vietnamese** only when:
  1. The student speaks Vietnamese (detect automatically)
  2. The student explicitly asks for explanation in Vietnamese (e.g. "giải thích", "nghĩa là gì", "tiếng việt")
- After a Vietnamese explanation, gently return to Japanese in the next turn.

## Response format
Always reply as a JSON object with this exact structure — no extra text outside JSON:
{{
  "reply_text": "The full reply to DISPLAY. Japanese with furigana hints if needed, or Vietnamese if explaining.",
  "tts_text": "The Japanese text to READ ALOUD. If reply is in Vietnamese, compose a short relevant Japanese sentence. Never empty.",
  "romaji": "Romaji of tts_text",
  "translation": "Vietnamese meaning of tts_text"
}}

## Level adjustments
{level_instruction}

## Style
- Warm, encouraging, like a real private tutor
- Use natural Japanese (not textbook stiff)
- Correct mistakes gently: repeat the correct form in Japanese
- Ask follow-up questions in Japanese to keep the conversation going
- Use furigana hints in reply_text for hard kanji: e.g. 食べ物（たべもの）
"""

LEVEL_INSTRUCTIONS = {
    "beginner":     "Complete beginner. Use ONLY hiragana + very simple words (JLPT N5). Always add romaji for every word in reply_text. Speak very slowly and simply.",
    "elementary":   "Basic Japanese (JLPT N5-N4). Simple kanji with furigana. Short sentences.",
    "intermediate": "Intermediate (JLPT N3-N4). Natural Japanese, kanji with occasional furigana. Normal pace.",
    "advanced":     "Advanced (JLPT N1-N2). Natural, nuanced Japanese. Minimal furigana. Complex topics OK.",
}

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)

@app.route("/chat", methods=["POST"])
def chat():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "MISTRAL_API_KEY not set"}), 500

    data    = request.json
    history = data.get("history", [])
    level   = data.get("level", "beginner")

    system   = SYSTEM_PROMPT.format(level_instruction=LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["beginner"]))
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
    if not ELEVENLABS_VOICE_ID:
        return jsonify({"error": "ELEVENLABS_VOICE_ID not set"}), 500

    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400

    resp = requests.post(
        f"{ELEVENLABS_URL}/{ELEVENLABS_VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        },
        json={
            "text": text,
            "model_id": "eleven_turbo_v2_5",  # best model for Japanese
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.3,
                "use_speaker_boost": True
            }
        }
    )

    if not resp.ok:
        return jsonify({"error": resp.text}), resp.status_code

    return Response(resp.content, mimetype="audio/mpeg")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)