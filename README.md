
### Instructions
1.  Create a file named `README.md` in your project root.
2.  Paste the content below.

***

```markdown
# Epistemiq: The Voice-Driven Scientific Truth Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4)
![ElevenLabs](https://img.shields.io/badge/Voice-ElevenLabs-black)

**Your Compass in the Epistemic Fog.**

Epistemiq is a voice-enabled cognitive agent that instantly verifies spoken claims against millions of scientific papers. It combines **Google Gemini's reasoning**, **Google's Semantic Embeddings**, and **ElevenLabs' conversational voice AI** to turn complex fact-checking into a simple conversation.

---

## 🤖 Features

*   **🎙️ Voice-to-Verdict:** Speak a claim (e.g., *"I heard scientists found life on Mars"*), and the agent processes it instantly.
*   **🧠 Deep Reasoning:** Uses **Google Gemma** for extraction and **Gemini** for complex scientific verification.
*   **📚 RAG Pipeline:** Retrieves real-time data from **Semantic Scholar, PubMed, CORE, and CrossRef**.
*   **🔍 Semantic Reranking:** Uses **Google `text-embedding-004`** with `FACT_VERIFICATION` task type to rank papers by relevance, not just keywords.
*   **🗣️ Audio Briefing:** Converts the verdict into a natural, podcast-style audio report using **ElevenLabs Turbo v2.5**.
*   **💬 Conversational Follow-up:** The agent listens for your response (e.g., "Tell me more about option 1") and generates deep-dive reports on the fly.

---

## 🛠️ Architecture & Tech Stack

Epistemiq is a Flask application backed by a sophisticated RAG (Retrieval-Augmented Generation) pipeline.

*   **Frontend:** HTML5, Bootstrap 5, Vanilla JS, Web Speech API (STT).
*   **Backend:** Python, Flask, Gunicorn (Threaded).
*   **Database:** PostgreSQL (Neon) with `pgvector` extension.
*   **AI Models:**
    *   **Reasoning:** gemma-3-27b-it & Gemini (via Google AI Studio).
    *   **Embeddings:** Google Vertex AI `text-embedding-004`.
    *   **Voice:** ElevenLabs API (`eleven_turbo_v2`).
*   **Infrastructure:** Docker, PythonAnywhere.

---

## 🚀 Getting Started

### Prerequisites

You need API keys for the following services:
1.  **Google AI Studio:** `GOOGLE_API_KEY`
2.  **ElevenLabs:** `ELEVENLABS_API_KEY`
3.  **Database:** A PostgreSQL database URL (Must support `vector` extension).
4.  **OpenRouter:** `OPENROUTER_API_KEY` (Used for fallback redundancy).
5.  **SendGrid:** (Optional) For magic link authentication.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/epistemiq.git
cd epistemiq
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```ini
# Database
DATABASE_URL=postgresql://user:password@host:port/dbname?sslmode=require

# AI & Voice Keys
GOOGLE_API_KEY=your_google_key
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENROUTER_API_KEY=your_openrouter_key

# App Security
FLASK_SECRET_KEY=super_secret_key

# Research APIs (Optional but recommended)
SEMANTIC_SCHOLAR_API_KEY=your_key
CORE_API_KEY=your_key

# Admin & Email
ADMIN_EMAILS=["your_email@example.com"]
SENDGRID_API_KEY=your_sendgrid_key
```

### 3. Run with Docker (Recommended)

The project includes a production-ready Docker setup.

```bash
# Build and run
docker-compose up --build
```

Access the app at `http://localhost:8080`.

### 4. Manual Installation

If you prefer running without Docker:

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python flask_app.py
```

---

## 🧠 How the RAG Pipeline Works

1.  **Extraction:** The user's input (Text or Speech) is passed to `Gemma` to extract testable claims.
2.  **Search:** We query external APIs (Semantic Scholar, PubMed) to fetch 20+ candidate papers.
3.  **Vector Reranking:**
    *   We generate a 768-dim vector for the Claim.
    *   We generate vectors for all paper abstracts.
    *   We calculate **Cosine Similarity** to rank papers by *meaning*, not just keywords.
4.  **Verification:** The top 6 papers are fed into `Gemini`, which acts as the Judge to issue a verdict.
5.  **Audio Synthesis:** The final text is streamed to ElevenLabs to generate the voice response.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
```
