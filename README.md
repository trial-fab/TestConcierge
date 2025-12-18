## USF Campus Concierge
AI concierge for Admissions and Orientation teams. Streamlit UI sits on top of a Supabase + pgvector RAG stack, Phi-4 handles chat/email/meeting flows, and Splunk provides observability.

---

### What’s Inside
- **Campus-aware chat** – Hugging Face `google/embeddinggemma-300m` embeddings land in Supabase pgvector; every response cites the exact markdown chunk.
- **Mini-assistants** – Email and Meeting builders run through the MCP server so drafts, edits, Gmail sends, and Calendar events stay auditable.
- **Safety + Auth** – Prompt-injection detector, token budgets, Supabase RLS per user, structured error handling.
- **Observability** – Streamlit + MCP + RAG + Google API events streamed to Splunk (dashboards + alerts included).

---

### Stack
| Area | Tech |
| --- | --- |
| Frontend | Streamlit, custom CSS |
| LLMs | Azure OpenAI Phi-4 orchestrator, email, meeting deployments |
| Embeddings | Hugging Face hosted inference (`google/embeddinggemma-300m`) |
| Backend | Supabase Postgres + pgvector + RPC for retrieval |
| Agent tooling | MCP server/client (`agents/mcp.py`) |
| Email/Calendar | Gmail + Google Calendar APIs |
| Observability | Splunk HEC (system health + RAG dashboards, alerts) |

---

### Screenshots
| Chat + Assistants | Recent Actions |
| --- | --- |
| <img width="2027" height="932" alt="Chat + assistants UI" src="https://github.com/user-attachments/assets/f16f24d9-0292-4639-879b-4bd4ae8cfaa9" /> | <img width="2546" height="1149" alt="Recent actions panel" src="https://github.com/user-attachments/assets/5a8c8e4d-c111-4141-ae84-ee74b9f57e8d" /> |

---

### Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in Azure/Supabase/HF/Google/Splunk creds
streamlit run app.py
```
Key env vars:
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_PHI4_*`
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- `HUGGINGFACEHUB_API_TOKEN`
- `GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN`
- `SPLUNK_HEC_URL`, `SPLUNK_HEC_TOKEN`, `SPLUNK_ENABLED=true`

Supabase prep: run schema in `docs/setup_guide.md`, enable RLS, then load markdown with `python setup_db.py --source data/raw`.

---

### Using the App
1. Register/login (Supabase stores users + sessions).
2. Chat – ask policy questions, each answer lists `Source N` links.
3. Tap **＋** to open assistants:
   - **Email**: enter student details → Generate Draft → optional AI edit → send via Gmail.
   - **Meeting**: summary/date/attendees → check availability → create event with Meet link.
4. Rename/export/delete sessions from sidebar; recent actions collapse into audit cards.

---

### Observability
- **Dashboards**: System Health + RAG Performance panels covering request health, pipeline times, and assistant usage.
- **Alerts**: “High Request Error Rate” and “RAG Latency Degradation” alerts hook into the same telemetry.
- **Logging**: `utils/splunk_logger.py` batches events, retries on failure, and falls back to `logs/splunk_fallback.log` if HEC is down.
Example searches:
```spl
index=main sourcetype="usf_concierge:request" event_type=request_error
| stats count by session_id

index=main sourcetype="usf_concierge:rag" event_type=vector_search
| stats avg(metrics.duration_ms) by _time span=5m
```
More detail lives in `docs/SPLUNK_QUICKSTART.md`, `docs/SPLUNK_QUERIES.md`, and the in-depth OBSERVABILITY doc referenced there.

---

### Keep the app up
`.github/workflows/keep-alive.yml` runs every 2 hours:
- Hits your Streamlit Cloud URL (set `STREAMLIT_APP_URL` in GitHub Actions secrets).
- Optionally pings Supabase REST (`SUPABASE_URL`, `SUPABASE_ANON_KEY` secrets).

---

### Roadmap
1. Per-user Gmail/Calendar OAuth instead of a shared service account.
2. Scheduled data ingestion with diffing + eval harness for RAG quality.
3. User-level audit exports + admin panel.

---

### Credits
- Fabrizio – RAG pipeline, Azure integration, assistants, Google tooling, ingestion, Splunk observability.
- Chi – UI/UX, Streamlit components, styling.
- Gang – Architecture, database schema, session/state/auth layers, security.
- Rishi – MCP server/client, tool orchestration, email/meeting workflow wiring.