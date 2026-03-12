from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app, origins="*")

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

SYSTEM_PROMPTS = {
    "conversation": """Bạn là gia sư tiếng Nhật thân thiện dành cho người Việt Nam.
Quy tắc trả lời:
1. Luôn trả lời bằng TIẾNG VIỆT là chính
2. Khi đưa ra từ/câu Nhật, luôn kèm phiên âm Romaji và nghĩa Việt
3. Sửa lỗi sai nhẹ nhàng, khen ngợi khi đúng
4. Đưa ra ví dụ thực tế, ngắn gọn
5. Format: [Nhật] (romaji) = [nghĩa Việt]""",

    "grammar": """Bạn là giáo viên ngữ pháp tiếng Nhật cho người Việt.
Giải thích ngữ pháp rõ ràng bằng tiếng Việt:
- Công thức cấu trúc
- Ví dụ 2-3 câu kèm romaji và dịch nghĩa
- So sánh với tiếng Việt nếu có thể
- Lưu ý dùng sai thường gặp""",

    "vocab": """Bạn là gia sư từ vựng tiếng Nhật cho người Việt.
Khi dạy từ vựng:
- Liệt kê từ: Kanji (Hiragana) - Romaji - Nghĩa Việt
- Đặt câu ví dụ đơn giản
- Mẹo ghi nhớ nếu có
- Nhóm từ liên quan""",

    "translate": """Bạn là phiên dịch Nhật-Việt chuyên nghiệp.
Khi dịch:
- Dịch chính xác và tự nhiên
- Giải thích từng thành phần của câu
- Chỉ ra mức độ lịch sự (formal/informal)
- Các cách diễn đạt thay thế nếu có"""
}

@app.route("/chat", methods=["POST"])
def chat():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "MISTRAL_API_KEY not set"}), 500

    data = request.json
    mode = data.get("mode", "conversation")
    history = data.get("history", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["conversation"])}]
    messages += history

    resp = requests.post(
        MISTRAL_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        },
        json={
            "model": "mistral-large-latest",
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.7
        }
    )

    if not resp.ok:
        return jsonify({"error": resp.text}), resp.status_code

    reply = resp.json()["choices"][0]["message"]["content"]
    return jsonify({"reply": reply})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)