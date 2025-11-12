## USF Onboarding Assistant
Streamlit concierge for Admissions/Orientation staff that grounds every answer in the USF onboarding corpus, drafts/sends emails, and books calendar invites with Google Meet links.

---

### Key Features (Advanced Techniques)
- **RAG over Supabase pgvector** – Markdown corpus chunked/embedded with Hugging Face (`google/embeddinggemma-300m`) and retrieved via a Supabase RPC before every response.
- **Azure Phi-4 Multi-Model Stack** – Three dedicated deployments (orchestrator, email specialist, meeting specialist) all accessed via Azure OpenAI so every workflow stays on Phi-4 while allowing task-specific prompting.
- **Real MCP Tooling** – Model Context Protocol server mediates every Gmail/Calendar, drafting, and planning action (`retrieve_context`, `draft_email`, `plan_meeting`, `send_email`, `create_event`, etc.) with structured audit logs.
- **Mini Assistants** – Streamlit chat page includes Email/Meeting builders that stream drafts, apply AI edits, send messages, and log recent actions.
- **Safety & Guardrails** – Prompt-injection detector, token budgeting, citations enforced by the system prompt, and Supabase RLS for per-user chat state.

---

### Screenshots
| Chat + Assistants | Recent Actions Log |
|-------------------|--------------------|
| ![Chat UI](docs/screenshots/chat-ui.png) | ![Actions](docs/screenshots/actions.png) |

---

### Setup Instructions
1. **Clone & Install**
   ```bash
   git clone <repo-url>
   cd TA
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure Environment**
   - Copy `.env.example` → `.env`.
   - Fill in Azure OpenAI (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`) plus the Phi-4 deployment names `AZURE_PHI4_ORCHESTRATOR`, `AZURE_PHI4_EMAIL`, `AZURE_PHI4_MEETING` (email/meeting fall back to the orchestrator name if omitted).
   - Provide Hugging Face, Supabase, and Google OAuth values (same keys go into Streamlit Cloud secrets).
3. **Prep Supabase**
   - Create tables/RPC from `docs/schema.sql` (or Supabase SQL editor) and enable RLS (policies described in `docs/setup_guide.md`).
4. **Ingest Content**
   ```bash
   python data_ingestion.py --source data/raw
   ```
   (Deletes and reloads `documents` + `chunks` each run.)
5. **Run Locally**
   ```bash
   streamlit run app.py
   ```
   (The MCP server auto-launches when the app invokes a tool. You can also run it manually with `python -m utils.mcp serve`.)
6. **Deploy**
   - Push to GitHub, set secrets in Streamlit Cloud, deploy `app.py`.

---

### How to Use
1. Register/login (accounts stored in Supabase).
2. Start a chat: ask policy questions; responses cite `Source N`.
3. Use **＋** to open assistants:
   - **Email Assistant** – enter student details → Generate Draft → Apply AI edit or Send.
   - **Meeting Assistant** – supply summary, date/time, attendees → Check availability → Create Event (Meet link returned).
4. Download/export sessions or rename/delete via the Options popover.
5. Review recent assisted actions in the chat history for auditing.

---

### Team Contributions
- **Alice Doe** – Streamlit UI/UX, assistant flows, MCP integration.
- **Bob Smith** – RAG pipeline, Supabase schema/RLS, deployment automation.
- _Update with actual team names/roles._

---

### Technologies
- **LLM**: Azure OpenAI Phi-4 (chat)
- **Embeddings**: Hugging Face `google/embeddinggemma-300m`
- **Vector Store & Auth**: Supabase Postgres + pgvector + RLS
- **Frontend**: Streamlit
- **Agent Tooling**: Model Context Protocol client/server (`utils/mcp.py`)
- **Email/Calendar**: Gmail + Google Calendar APIs (OAuth refresh flow)

---

### Known Limitations
- Corpus limited to curated Markdown set; if content is missing, the bot must refuse.
- Single service account handles Gmail/Calendar actions; no per-user delegation yet.
- Google Meet links issued only when the Calendar API token has `calendar.events` scope.
- No offline/queueing—assistants require live API access.

---

### Future Improvements
1. Add per-user OAuth so assistants act on behalf of individual staff.
2. Expand corpus ingestion pipeline with auto-diff + scheduled refreshes.
3. Build evaluation harness for RAG answer quality + guardrail coverage.

---

### Additional Docs
- `docs/architecture_diagram.png` – System + data flow.
- `docs/agent_workflow.png` – RAG + assistant tool sequence.
- `docs/setup_guide.md` – Supabase schema, RLS policies, deployment checklist.

---

### Supabase Schema (Quick Reference)
```sql
create extension if not exists vector;

create table if not exists users (
  id uuid primary key,
  username text unique not null,
  email text,
  salt text not null,
  pwd_hash text not null,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists chat_sessions (
  id uuid primary key,
  user_id uuid references users(id) on delete cascade,
  session_name text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists messages (
  id uuid primary key,
  session_id uuid references chat_sessions(id) on delete cascade,
  role text,
  content text,
  tokens_in int,
  tokens_out int,
  created_at timestamptz default now()
);

create table if not exists audit_logs (
  id uuid primary key,
  session_id uuid references chat_sessions(id) on delete cascade,
  event_type text,
  payload jsonb,
  created_at timestamptz default now()
);

create table if not exists documents (
  id uuid primary key,
  title text,
  source_path text,
  category text,
  checksum text unique,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists chunks (
  id uuid primary key,
  document_id uuid references documents(id) on delete cascade,
  chunk_index int,
  content text,
  section_title text,
  metadata jsonb,
  embedding vector(3072),
  chunk_fp text unique,
  created_at timestamptz default now()
);

create or replace function match_document_chunks(
  query_embedding vector(3072),
  match_count int default 6,
  filter jsonb default '{}'::jsonb
)
returns table (
  id uuid,
  content text,
  section_title text,
  metadata jsonb,
  similarity double precision
) language plpgsql as $$
begin
  return query
  select
    c.id,
    c.content,
    c.section_title,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from chunks c
  where (filter = '{}'::jsonb) or (c.metadata @> filter)
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;
```
`data_ingestion.py` performs a full refresh each run by deleting all rows in `documents` and `chunks` before inserting the newly embedded corpus, so the tables stay clean without manual SQL.
