from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json
import re

app = Flask(__name__)
CORS(app, origins="*")

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
GOOGLE_TTS_KEY  = os.environ.get("GOOGLE_TTS_KEY", "")
MISTRAL_URL     = "https://api.mistral.ai/v1/chat/completions"
GOOGLE_TTS_URL  = "https://texttospeech.googleapis.com/v1/text:synthesize"

# Google TTS voice options
VOICES = {
    "female": "ja-JP-Neural2-B",
    "male":   "ja-JP-Neural2-C",
}

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
  "reply_text": "The full reply to DISPLAY (mix of Japanese + romaji hints as needed). If replying in Vietnamese, write Vietnamese here.",
  "tts_text": "The Japanese text to READ ALOUD via TTS. If the reply is fully Vietnamese, extract or compose a short relevant Japanese sentence/phrase to read. Never leave this empty.",
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
    "beginner":       "Student is a complete beginner. Use ONLY hiragana + very simple words (JLPT N5). Always add romaji for every Japanese word in reply_text. Speak very slowly and simply.",
    "elementary":     "Student knows basic Japanese (JLPT N5-N4). Use simple kanji with furigana hints. Keep sentences short.",
    "intermediate":   "Student is intermediate (JLPT N3-N4). Use natural Japanese, kanji with occasional furigana for harder words. Normal pace.",
    "advanced":       "Student is advanced (JLPT N1-N2). Use natural, nuanced Japanese. Minimal furigana. Can discuss complex topics.",
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

    level_instruction = LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["beginner"])
    system = SYSTEM_PROMPT.format(level_instruction=level_instruction)

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


# TTS is now handled entirely in the browser via Web Speech API (ja-JP)
# No backend TTS endpoint needed


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)