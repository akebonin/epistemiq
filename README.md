# **Epistemiq — Hybrid Local + Cloud AI Fact-Checking with Chrome Built-in AI + Whisper WASM**

**Epistemiq** is a browser-first tool for rapid, privacy-enhanced misinformation analysis. It blends **local AI** with **controlled, cached cloud reasoning** to extract claims, verify them, and generate reports.

### ✅ Core AI Approach

| Function                                        | Runs Where              | Tech                                       |
| ----------------------------------------------- | ----------------------- | ------------------------------------------ |
| Text summarization                              | **Local**               | Chrome Built-in AI (Gemini Nano)           |
| Speech-to-text                                   | **Local**               | Whisper tiny.en (Transformers.js WASM)     |
| Claim extraction, verification & report writing | **Cloud + caching**     | OpenRouter LLM (Mistral / Gemini / Claude) |
                          |

> Cloud LLM calls are **cached, deduplicated, and minimized** using text hashing.
> When possible, preprocessing happens **entirely on-device**.

---

## 🔐 Privacy-Forward Design

| Layer                                       | Privacy Guarantee                                        |
| ------------------------------------------- | -------------------------------------------------------- |
| Audio transcription                         | **Local** — no audio leaves the browser                  |
| Text preprocessing (if Chrome AI available) | **Local** — runs in browser memory                       |
| Claim extraction & research flow            | **Cloud call + persistent cache** to avoid reruns        |
| Image processing                            | Backend only fetches media → browser transcribes locally |

Epistemiq reduces sensitive data exposure by:

* Performing speech-to-text locally
* Using Nano for local summarization before any cloud call
* Hashing text → **reuse past results** instead of repeated model queries

---

## 🚀 Features

| Capability                     | Tech Used                                      |
| ------------------------------ | ---------------------------------------------- |
| Text → extract claims & verify | OpenRouter LLM (cached) + Chrome AI (optional) |
| Summarize before analyzing     | `window.ai.prompt()` → Gemini Nano             |
| Local speech-to-text           | Whisper WASM                                   |
| PDF reports                    | Browser-generated                              |
| Caching across sessions        | SQLite + text hashing                          |

---

## 🧠 Why It Matters

Misinformation defense requires **speed, trust, and accessibility**.

Epistemiq demonstrates:

* Browser-native AI flows
* Hybrid models that **prioritize local first, cloud only when needed**
* Educational and civic fact-checking tool design
* True **Edge + Cloud cooperation**, not cloud-dependence

Think:

> “AI-augmented critical thinking, not AI-powered hallucination.”

---

## 🛠 Tech Stack

### Frontend

* HTML / CSS / Bootstrap
* Pure JavaScript
* Chrome Built-in AI (`window.ai.prompt`)
* Transformers.js + WASM Whisper
* Service Worker (offline caching)

### Backend

* Flask (PythonAnywhere)
* `yt-dlp` for media fetch
* `ffmpeg` for 16kHz WAV export
* SQLite caching

---

## 🧪 Functionality Notes

* Chrome Nano mode = **private preprocessing**
* No Chrome AI? → **still works** (cloud + cache fallback)
* Whisper STT always **runs locally**
* Cached results bypass rate limits & latency entirely

---

## ✅ Live Demo

Frontend: [https://epistemiq.vercel.app](https://epistemiq.vercel.app)
Backend: PythonAnywhere (API)
Repo: *(this repo)*

**Chrome Canary recommended for local-AI mode**

---

No cloud speech API.
Full local voice processing.

---

## 📦 Project Structure

```
frontend/
backend/
static/
templates/
sessions.db (auto-recreated)
```

---

## 🧠 Architecture (Simplified)

```
Browser:
  Nano → Summarization
  Whisper → STT
  UI + PDF export
  Caching logic (hash check)

Backend:
  yt-dlp + ffmpeg
  SQLite cache for LLM calls

Cloud:
  OpenRouter LLM (claims assessment + reports)
```

---

## 🧾 License

MIT — extend freely.

---

## 👋 Contact

Built by **Asparuh Kebonin / Alis Grave Nil**
Hackathon entry: Chrome Built-in AI Challenge

---

### ✨ Final Notes for Judges

This project demonstrates:

* Practical **local AI deployment**
* Real browser-side inference
* Sensible hybrid compute for real-world scale
* Privacy-aware design (no raw audio to cloud)
* Persistent caching to minimize cost + API use

**The goal is not to replace fact-checkers —
but to empower users with faster truth-seeking tools.**
