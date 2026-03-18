import Fastify from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import Database from "better-sqlite3";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const JWT_SECRET          = process.env.JWT_SECRET ?? "dev_secret_change_in_prod";
const FREE_DAILY_SECONDS  = 600; // 10 minutes
const MISTRAL_API_KEY     = process.env.MISTRAL_API_KEY    ?? "";
const MISTRAL_URL         = "https://api.mistral.ai/v1/chat/completions";
const ELEVENLABS_API_KEY  = process.env.ELEVENLABS_API_KEY ?? "";
const ELEVENLABS_URL      = "https://api.elevenlabs.io/v1/text-to-speech";

const VOICE_IDS = {
  jf_alpha:      process.env.VOICE_JF_ALPHA      ?? "EXAVITQu4vr4xnSDxMaL",
  jf_gongitsune: process.env.VOICE_JF_GONGITSUNE ?? "9BWtsMINqrJLrRacOk9x",
  jm_kumo:       process.env.VOICE_JM_KUMO       ?? "IKne3meq5aSn9XLyUdCD",
};
const DEFAULT_VOICE = "jf_alpha";
const LANG_NAMES = { vi: "Vietnamese", en: "English", ja: "Japanese" };

// ── Database ──────────────────────────────────────────────────────────────────
const db = new Database(process.env.DB_PATH ?? path.join(__dirname, "data.db"));
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_premium    INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS daily_usage (
    user_id INTEGER NOT NULL,
    date    TEXT    NOT NULL,
    seconds INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(id)
  );
`);

// ── Fastify ───────────────────────────────────────────────────────────────────
const app = Fastify();
await app.register(cors, {
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
});
await app.register(multipart);

// ── Auth helpers ──────────────────────────────────────────────────────────────
const signToken = (user) =>
  jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: "30d" });

const getAuthUser = (req) => {
  const auth = req.headers.authorization;
  if (!auth?.startsWith("Bearer ")) return null;
  try { return jwt.verify(auth.slice(7), JWT_SECRET); }
  catch { return null; }
};

const requireAuth = (req, reply) => {
  const u = getAuthUser(req);
  if (!u) { reply.code(401).send({ error: "Unauthorized" }); return null; }
  return u;
};

// ── Usage helpers ─────────────────────────────────────────────────────────────
const today   = () => new Date().toISOString().slice(0, 10);
const getUsed = (uid) =>
  db.prepare("SELECT seconds FROM daily_usage WHERE user_id=? AND date=?").get(uid, today())?.seconds ?? 0;
const addUsed = (uid, secs) =>
  db.prepare(`INSERT INTO daily_usage (user_id,date,seconds) VALUES (?,?,?)
    ON CONFLICT(user_id,date) DO UPDATE SET seconds=seconds+?`
  ).run(uid, today(), secs, secs);

// ── System prompt ─────────────────────────────────────────────────────────────
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

const LEVEL_INSTRUCTIONS = {
  beginner:     "Learner is a complete beginner (JLPT N5). Use simple Japanese but NEVER sacrifice politeness for simplicity — staff/service roles must still use proper keigo (いらっしゃいませ、何になさいますか、～でございます). Add furigana for all kanji in reply_text.",
  elementary:   "Learner is elementary (JLPT N5-N4). Use simple natural Japanese. Service roles use proper keigo. Add furigana for harder kanji.",
  intermediate: "Learner is intermediate (JLPT N3-N4). Use natural Japanese appropriate to the role. Service roles use keigo, friends use casual speech. Furigana only for N2+ kanji.",
  advanced:     "Learner is advanced (JLPT N1-N2). Use fully natural Japanese. Keigo for formal roles, casual/slang for friend roles. No furigana needed.",
};

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

// ── Auth routes ───────────────────────────────────────────────────────────────
app.post("/auth/register", async (req, reply) => {
  const { email, password } = req.body ?? {};
  if (!email || !password) return reply.code(400).send({ error: "Email and password required" });
  if (password.length < 6) return reply.code(400).send({ error: "Password min 6 characters" });

  const hash = await bcrypt.hash(password, 10);
  try {
    const { lastInsertRowid } = db.prepare(
      "INSERT INTO users (email, password_hash) VALUES (?,?)"
    ).run(email.toLowerCase().trim(), hash);
    const user = db.prepare("SELECT * FROM users WHERE id=?").get(lastInsertRowid);
    return { token: signToken(user), user: { id: user.id, email: user.email, is_premium: !!user.is_premium } };
  } catch (e) {
    if (e.message?.includes("UNIQUE")) return reply.code(409).send({ error: "Email already registered" });
    throw e;
  }
});

app.post("/auth/login", async (req, reply) => {
  const { email, password } = req.body ?? {};
  const user = db.prepare("SELECT * FROM users WHERE email=?").get(email?.toLowerCase().trim());
  if (!user || !(await bcrypt.compare(password ?? "", user.password_hash)))
    return reply.code(401).send({ error: "Invalid email or password" });
  return { token: signToken(user), user: { id: user.id, email: user.email, is_premium: !!user.is_premium } };
});

app.get("/auth/me", (req, reply) => {
  const auth = requireAuth(req, reply);
  if (!auth) return;
  const user = db.prepare("SELECT * FROM users WHERE id=?").get(auth.id);
  if (!user) return reply.code(404).send({ error: "User not found" });
  return {
    user: { id: user.id, email: user.email, is_premium: !!user.is_premium },
    usage: { used: getUsed(user.id), limit: FREE_DAILY_SECONDS },
  };
});

// ── Payment ───────────────────────────────────────────────────────────────────
app.post("/payment/activate", (req, reply) => {
  const auth = requireAuth(req, reply);
  if (!auth) return;
  db.prepare("UPDATE users SET is_premium=1 WHERE id=?").run(auth.id);
  const user = db.prepare("SELECT * FROM users WHERE id=?").get(auth.id);
  return {
    success: true,
    token: signToken(user),
    user: { id: user.id, email: user.email, is_premium: true },
  };
});

// ── Chat ──────────────────────────────────────────────────────────────────────
app.post("/chat", async (req, reply) => {
  if (!MISTRAL_API_KEY) return reply.code(500).send({ error: "MISTRAL_API_KEY not set" });
  const auth = requireAuth(req, reply);
  if (!auth) return;

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

// ── TTS ───────────────────────────────────────────────────────────────────────
app.post("/tts", async (req, reply) => {
  if (!ELEVENLABS_API_KEY) return reply.code(500).send({ error: "ELEVENLABS_API_KEY not set" });
  const auth = requireAuth(req, reply);
  if (!auth) return;

  const user = db.prepare("SELECT * FROM users WHERE id=?").get(auth.id);
  if (!user.is_premium) {
    const used = getUsed(auth.id);
    if (used >= FREE_DAILY_SECONDS)
      return reply.code(402).send({ error: "limit_reached", used, limit: FREE_DAILY_SECONDS });
  }

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

  if (!user.is_premium) addUsed(auth.id, Math.ceil(text.length / 3));

  reply.header("Content-Type", "audio/mpeg");
  return reply.send(Buffer.from(await res.arrayBuffer()));
});

// ── STT ───────────────────────────────────────────────────────────────────────
app.post("/stt", async (req, reply) => {
  if (!ELEVENLABS_API_KEY) return reply.code(500).send({ error: "ELEVENLABS_API_KEY not set" });
  const auth = requireAuth(req, reply);
  if (!auth) return;

  const user = db.prepare("SELECT * FROM users WHERE id=?").get(auth.id);
  if (!user.is_premium) {
    const used = getUsed(auth.id);
    if (used >= FREE_DAILY_SECONDS)
      return reply.code(402).send({ error: "limit_reached", used, limit: FREE_DAILY_SECONDS });
  }

  let audioBuffer, audioMime, langCode = "ja";
  for await (const part of req.parts()) {
    if (part.type === "file" && part.fieldname === "file") {
      audioBuffer = await part.toBuffer();
      audioMime   = part.mimetype || "audio/webm";
    } else if (part.type === "field" && part.fieldname === "lang") {
      langCode = part.value || "ja";
    }
  }
  if (!audioBuffer) return reply.code(400).send({ error: "no audio" });

  const form = new FormData();
  form.append("file", new Blob([audioBuffer], { type: audioMime }), "audio.webm");
  form.append("model_id", "scribe_v1");
  form.append("language_code", langCode);

  const res = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
    method: "POST",
    headers: { "xi-api-key": ELEVENLABS_API_KEY },
    body: form,
  });
  if (!res.ok) return reply.code(res.status).send({ error: await res.text() });

  if (!user.is_premium) addUsed(auth.id, 10);

  const result = await res.json();
  return { text: result.text ?? "" };
});

// ── Analyze ───────────────────────────────────────────────────────────────────
app.post("/analyze", async (req, reply) => {
  if (!MISTRAL_API_KEY) return reply.code(500).send({ error: "MISTRAL_API_KEY not set" });
  const auth = requireAuth(req, reply);
  if (!auth) return;

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

// ── Translate ─────────────────────────────────────────────────────────────────
app.post("/translate", async (req, reply) => {
  if (!MISTRAL_API_KEY) return reply.code(500).send({ error: "MISTRAL_API_KEY not set" });
  const auth = requireAuth(req, reply);
  if (!auth) return;

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

// ── Health ────────────────────────────────────────────────────────────────────
app.get("/health", () => ({ status: "ok", tts: "elevenlabs", stt: "elevenlabs" }));

// ── Start ─────────────────────────────────────────────────────────────────────
const port = parseInt(process.env.PORT ?? "5000");
await app.listen({ host: "0.0.0.0", port });
console.log(`Server running on port ${port}`);
