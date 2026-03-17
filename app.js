import Fastify from "fastify";
import cors from "@fastify/cors";

const app = Fastify();
await app.register(cors, { origin: "*" });

const MISTRAL_API_KEY    = process.env.MISTRAL_API_KEY    ?? "";
const MISTRAL_URL        = "https://api.mistral.ai/v1/chat/completions";
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY ?? "";
const ELEVENLABS_URL     = "https://api.elevenlabs.io/v1/text-to-speech";

const VOICE_IDS = {
  jf_alpha:      process.env.VOICE_JF_ALPHA      ?? "EXAVITQu4vr4xnSDxMaL",
  jf_gongitsune: process.env.VOICE_JF_GONGITSUNE ?? "9BWtsMINqrJLrRacOk9x",
  jm_kumo:       process.env.VOICE_JM_KUMO       ?? "IKne3meq5aSn9XLyUdCD",
};
const DEFAULT_VOICE = "jf_alpha";

const LANG_NAMES = { vi: "Vietnamese", en: "English", ja: "Japanese" };

const LEVEL_INSTRUCTIONS = {
  beginner:     "Learner is a complete beginner (JLPT N5). Use simple Japanese but NEVER sacrifice politeness for simplicity — staff/service roles must still use proper keigo (いらっしゃいませ、何になさいますか、～でございます). Add furigana for all kanji in reply_text.",
  elementary:   "Learner is elementary (JLPT N5-N4). Use simple natural Japanese. Service roles use proper keigo. Add furigana for harder kanji.",
  intermediate: "Learner is intermediate (JLPT N3-N4). Use natural Japanese appropriate to the role. Service roles use keigo, friends use casual speech. Furigana only for N2+ kanji.",
  advanced:     "Learner is advanced (JLPT N1-N2). Use fully natural Japanese. Keigo for formal roles, casual/slang for friend roles. No furigana needed.",
};

const SYSTEM_PROMPT = `You are playing a role in a Japanese conversation practice scenario with a learner.

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
{
  "reply_text": "What to DISPLAY. Japanese with furigana for hard kanji e.g. 注文（ちゅうもん）. Or {ui_lang_name} if explaining.",
  "tts_text": "Japanese to READ ALOUD. Must be natural spoken Japanese. Never empty.",
  "romaji": "Romaji of tts_text",
  "translation": "{ui_lang_name} meaning of tts_text"
}

## Level adjustments
{level_instruction}

## Keep the conversation alive — ALWAYS
NEVER end a response without driving the conversation forward. Always append a question or next action.
One response = action/statement + follow-up question. Never just a statement alone.

## Keigo (politeness) rules — STRICTLY follow
- Service staff (店員、受付、医者、面接官 etc.) → always use 敬語: ～でございます、～になさいますか、少々お待ちください、ありがとうございます
- Friends / classmates → casual: ～だよ、～じゃない？、ね、よ
- Never mix keigo and casual in the same role

## Important
- Keep responses SHORT and natural (1-3 sentences) like real conversation
- tts_text must be PURE Japanese with no romaji, brackets, or annotations`;

function buildSystemPrompt(level, uiLangName) {
  return SYSTEM_PROMPT
    .replaceAll("{ui_lang_name}", uiLangName)
    .replace("{level_instruction}", LEVEL_INSTRUCTIONS[level] ?? LEVEL_INSTRUCTIONS.beginner);
}

function extractJson(text) {
  text = text.trim().replace(/^```json|^```|```$/gm, "").trim();
  return JSON.parse(text);
}

async function mistral(messages, maxTokens = 600, temperature = 0.7) {
  const res = await fetch(MISTRAL_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${MISTRAL_API_KEY}` },
    body: JSON.stringify({ model: "mistral-large-latest", messages, max_tokens: maxTokens, temperature }),
  });
  if (!res.ok) throw { status: res.status, body: await res.text() };
  return (await res.json()).choices[0].message.content;
}

// ── Routes ────────────────────────────────────────────────────────────────────
app.post("/chat", async (req, reply) => {
  if (!MISTRAL_API_KEY) return reply.code(500).send({ error: "MISTRAL_API_KEY not set" });

  const { history = [], level = "beginner", ui_lang = "vi" } = req.body;
  const uiLangName = LANG_NAMES[ui_lang] ?? "Vietnamese";
  const messages = [{ role: "system", content: buildSystemPrompt(level, uiLangName) }, ...history];

  try {
    const raw = await mistral(messages);
    try { return extractJson(raw); }
    catch { return { reply_text: raw, tts_text: "", romaji: "", translation: "" }; }
  } catch (e) {
    return reply.code(e.status ?? 500).send({ error: e.body ?? String(e) });
  }
});

app.post("/tts", async (req, reply) => {
  if (!ELEVENLABS_API_KEY) return reply.code(500).send({ error: "ELEVENLABS_API_KEY not set" });

  const { text = "", voice = DEFAULT_VOICE } = req.body;
  if (!text.trim()) return reply.code(400).send({ error: "no text" });

  const voiceId = VOICE_IDS[voice] ?? VOICE_IDS[DEFAULT_VOICE];
  const res = await fetch(`${ELEVENLABS_URL}/${voiceId}`, {
    method: "POST",
    headers: { "xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      model_id: "eleven_turbo_v2_5",
      voice_settings: { stability: 0.5, similarity_boost: 0.75 },
    }),
  });

  if (!res.ok) return reply.code(res.status).send({ error: await res.text() });

  reply.header("Content-Type", "audio/mpeg");
  return reply.send(Buffer.from(await res.arrayBuffer()));
});

app.post("/analyze", async (req, reply) => {
  if (!MISTRAL_API_KEY) return reply.code(500).send({ error: "MISTRAL_API_KEY not set" });

  const { text = "", level = "beginner", ui_lang = "vi" } = req.body;
  if (!text.trim()) return reply.code(400).send({ error: "no text" });

  const uiLangName = LANG_NAMES[ui_lang] ?? "Vietnamese";
  const prompt = `Analyze this message from a Japanese learner (level: ${level}): "${text}"

Reply ONLY as JSON, no extra text:
{
  "is_japanese": true/false,
  "is_correct": true/false,
  "verdict": "one short sentence in ${uiLangName} summarizing correctness",
  "correction": "corrected Japanese sentence if wrong, else null",
  "tip": "one short ${uiLangName} tip about grammar/vocabulary/politeness if relevant, else null",
  "japanese": "if not Japanese input: how to say it in Japanese",
  "romaji": "romaji of japanese field if provided"
}

Be concise. verdict max 10 words. tip max 15 words.`;

  try {
    const raw = await mistral([{ role: "user", content: prompt }], 200, 0.3);
    return extractJson(raw);
  } catch (e) {
    return reply.code(e.status ?? 500).send({ error: e.body ?? "parse error" });
  }
});

app.post("/translate", async (req, reply) => {
  if (!MISTRAL_API_KEY) return reply.code(500).send({ error: "MISTRAL_API_KEY not set" });

  const { word = "", ui_lang = "vi" } = req.body;
  if (!word.trim()) return reply.code(400).send({ error: "no word" });

  const uiLangName = LANG_NAMES[ui_lang] ?? "Vietnamese";
  const prompt = `Translate this Japanese word or phrase to ${uiLangName}: "${word}"
Reply ONLY as JSON, no extra text:
{"translation": "concise ${uiLangName} meaning (max 8 words)", "romaji": "romaji reading"}`;

  try {
    const raw = await mistral([{ role: "user", content: prompt }], 100, 0.2);
    return extractJson(raw);
  } catch (e) {
    return reply.code(e.status ?? 500).send({ error: e.body ?? "parse error" });
  }
});

app.get("/health", () => ({ status: "ok", tts: "elevenlabs" }));

// ── Start ─────────────────────────────────────────────────────────────────────
const port = parseInt(process.env.PORT ?? "5000");
await app.listen({ host: "0.0.0.0", port });
console.log(`Server running on port ${port}`);
