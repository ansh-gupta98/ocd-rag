import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
SEVERITY_LEVELS  = {"LOW", "MILD", "HIGH"}
MAX_INPUT_CHARS  = 3500
HISTORY_WINDOW   = 15          # last N turns kept in rolling window

HF_LLM_REPO_ID  = os.getenv("HF_LLM_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct")
HF_EMBED_MODEL   = os.getenv("HF_EMBED_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")

llm        = None
embeddings = None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="OCD RAG Support API",
    description="Backend for OCD patient support chatbot. Consumed by Kotlin Retrofit.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas (Retrofit JSON contract) ─────────────────────────────────

class StartSessionResponse(BaseModel):
    session_id: str
    created_at: str

class ChatRequest(BaseModel):
    session_id: str                         = Field(..., description="UUID from /session/start")
    message: str                            = Field(..., description="User message text")
    kotlin_severity: Optional[str]          = Field(None, description="LOW | MILD | HIGH from Kotlin classifier")

class MessageItem(BaseModel):
    role: str          # "user" | "assistant"
    message: str
    severity: str
    timestamp: str

class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    ai_response: str
    severity_used: str
    severity_model: str
    severity_kotlin: Optional[str]
    timestamp: str

class SummaryRequest(BaseModel):
    session_id: str

class SummaryResponse(BaseModel):
    session_id: str
    generated_at: str
    summary_text: str
    event_count: int
    messages: List[MessageItem]


# ── Client init ───────────────────────────────────────────────────────────────

def _init_clients() -> None:
    global llm, embeddings
    if llm is not None and embeddings is not None:
        return

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "No HuggingFace token found. "
            "Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in your .env file."
        )

    endpoint = HuggingFaceEndpoint(
        repo_id=HF_LLM_REPO_ID,
        huggingfacehub_api_token=hf_token,
        task="conversational",
        max_new_tokens=512,
        temperature=0.4,
        do_sample=True,
    )
    llm = ChatHuggingFace(llm=endpoint)

    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_severity(raw: str) -> str:
    text = (raw or "").strip().upper()
    if "HIGH" in text:
        return "HIGH"
    if "MILD" in text:
        return "MILD"
    return "LOW"


def _knowledge_dir_fingerprint(knowledge_dir: Path) -> str:
    h = hashlib.sha256()
    if not knowledge_dir.is_dir():
        return ""
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        h.update(str(path.relative_to(knowledge_dir)).encode("utf-8", errors="replace"))
        h.update(str(stat.st_mtime_ns).encode("ascii"))
        h.update(str(stat.st_size).encode("ascii"))
    return h.hexdigest()


def _load_documents_from_directory(knowledge_dir: Path) -> List[Document]:
    docs: List[Document] = []
    if not knowledge_dir.is_dir():
        print(f"Warning: knowledge directory {knowledge_dir} does not exist.")
        return docs
    for f in knowledge_dir.rglob("*.txt"):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    for f in knowledge_dir.rglob("*.md"):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    for f in knowledge_dir.rglob("*.pdf"):
        print(f"Loading PDF {f}...")
        docs.extend(PyPDFLoader(str(f)).load())
    return docs


def _build_or_load_knowledge_faiss(knowledge_dir: Path, vector_dir: Path, emb) -> FAISS:
    raw_docs = _load_documents_from_directory(knowledge_dir)
    if not raw_docs:
        raise ValueError(
            f"No documentation files found under {knowledge_dir}. "
            "Add .txt, .md, or .pdf files (e.g. ocd_documentation/)."
        )

    splitter   = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    split_docs = splitter.split_documents(raw_docs)

    rebuild    = os.getenv("OCD_REBUILD_VECTOR", "").lower() in ("1", "true", "yes")
    meta_path  = vector_dir / "rag_meta.json"
    index_file = vector_dir / "index.faiss"
    emb_fp     = str(getattr(emb, "model_name", "unknown"))
    src_fp     = _knowledge_dir_fingerprint(knowledge_dir)

    if not rebuild and index_file.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("embedding_fingerprint") == emb_fp and meta.get("sources_fingerprint") == src_fp:
                return FAISS.load_local(str(vector_dir), emb, allow_dangerous_deserialization=True)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    vector_dir.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(split_docs, emb)
    db.save_local(str(vector_dir))
    meta_path.write_text(
        json.dumps({
            "embedding_fingerprint": emb_fp,
            "sources_fingerprint":   src_fp,
            "knowledge_dir":         str(knowledge_dir.resolve()),
            "chunk_count":           len(split_docs),
        }, indent=2),
        encoding="utf-8",
    )
    return db


# ── Shared chat helper ────────────────────────────────────────────────────────

def _chat(system: str, user: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content  = getattr(response, "content", str(response))
    if isinstance(content, list):
        return " ".join(str(x) for x in content)
    return str(content).strip()


def _policy_for_severity(severity: str) -> str:
    if severity == "LOW":
        return (
            "You may provide coping advice and practical self-help. "
            "Encourage optional check-in with a therapist if symptoms persist. "
            "Console the patient and tell them it's ok. "
            "Encourage small talks with friends/family and activities that bring joy."
        )
    if severity == "MILD":
        return (
            "Provide short coping suggestions but encourage meeting a licensed mental health "
            "professional soon — no pressure. Avoid presenting self-help as sufficient."
        )
    return (
        "Keep calm and supportive. Strongly advise urgent contact with a licensed mental health "
        "professional. If there is immediate risk or self-harm concern, advise emergency services. "
        "Treat with utmost caution and empathy. "
        "Provide Indian emergency numbers (iCall: 9152987821, Vandrevala: 1860-2662-345) if needed."
    )


# ── Service (singleton) ───────────────────────────────────────────────────────

class OCDRAGService:
    def __init__(self) -> None:
        _init_clients()
        root = Path(__file__).resolve().parent
        kd   = Path(os.getenv("OCD_KNOWLEDGE_DIR")    or root / "ocd_documentation")
        vs   = Path(os.getenv("OCD_VECTOR_STORE_DIR") or root / "ocd_documentation_vector")
        self.knowledge_db = _build_or_load_knowledge_faiss(kd, vs, embeddings)
        self.history_db   = FAISS.from_texts(["bootstrap memory"], embeddings)
        # session_id -> list of event dicts
        self.sessions: Dict[str, List[Dict]] = {}

    # ── session ──────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    # ── severity ─────────────────────────────────────────────────────────────

    def classify_severity(self, user_input: str) -> str:
        system = (
            "You are a strict mental health triage classifier for OCD support.\n"
            "Classify the user message severity as exactly one of: LOW, MILD, or HIGH.\n"
            "LOW  – minor intrusive thoughts, little functional impact.\n"
            "MILD – distress present, some functional impact, can still manage.\n"
            "HIGH – severe distress, impairment, safety risk, panic, or inability to function.\n"
            "Return EXACTLY one token: LOW or MILD or HIGH. No other text."
        )
        return _coerce_severity(_chat(system, user_input))

    # ── history rendering (last HISTORY_WINDOW turns) ────────────────────────

    def _render_recent_history(self, session_id: str, user_query: str) -> str:
        events = self.sessions.get(session_id, [])
        if not events:
            return "No prior turns."

        window = events[-HISTORY_WINDOW:]
        lines  = []
        for e in window:
            lines.append(f"user: {e['user']}")
            lines.append(f"assistant: {e['ai']}")

        # also pull semantically relevant older turns from FAISS history
        memory_hits = self.history_db.similarity_search(
            f"{session_id} | {user_query}", k=5,
            filter={"session_id": session_id}
        )
        history_text = "\n".join(lines)
        if memory_hits:
            history_text += "\n\nRelevant older memory:\n" + "\n".join(
                d.page_content for d in memory_hits
            )
        return history_text

    # ── chat ─────────────────────────────────────────────────────────────────

    def chat(
        self,
        session_id: str,
        user_input: str,
        kotlin_severity: Optional[str] = None,
    ) -> Dict:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Call /session/start first.")

        user_input     = user_input.strip()[:MAX_INPUT_CHARS]
        model_severity = self.classify_severity(user_input)
        final_severity = _coerce_severity(kotlin_severity) if kotlin_severity else model_severity

        context_docs = self.knowledge_db.as_retriever(search_kwargs={"k": 4}).invoke(user_input)
        context      = "\n".join(doc.page_content for doc in context_docs)
        history_text = self._render_recent_history(session_id, user_query=user_input)

        system = (
            "You are an OCD support assistant.\n"
            "You MUST use the provided Clinical Context to answer.\n"
            "If context is available, base your answer on it.\n\n"
            f"Clinical Context:\n{context}\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Severity: {final_severity}\n"
            f"Policy: {_policy_for_severity(final_severity)}\n\n"
            "Rules:\n"
            "- Always refer to context when possible\n"
            "- Be empathetic\n"
            "- No diagnosis or medication instructions\n"
            "- Max 150 words"
        )
        ai_text = _chat(system, user_input)

        event = {
            "timestamp":        _now_iso(),
            "session_id":       session_id,
            "user":             user_input,
            "ai":               ai_text,
            "severity":         final_severity,
            "severity_model":   model_severity,
            "severity_kotlin":  (kotlin_severity or "").upper(),
        }
        self.sessions[session_id].append(event)

        # persist to FAISS history for semantic recall
        self.history_db.add_texts(
            texts=[
                f"{event['timestamp']} | user: {event['user']}",
                f"{event['timestamp']} | assistant: {event['ai']}",
            ],
            metadatas=[
                {"session_id": session_id, "role": "user"},
                {"session_id": session_id, "role": "assistant"},
            ],
        )
        return event

    # ── summary ───────────────────────────────────────────────────────────────

    def summary_for_doctor(self, session_id: str) -> Dict:
        events = self.sessions.get(session_id, [])
        if not events:
            return {
                "session_id":   session_id,
                "generated_at": _now_iso(),
                "summary_text": "No conversation history available.",
                "event_count":  0,
                "messages":     [],
            }

        history_blob = "\n".join(
            f"{e['timestamp']} | user={e['user']} | severity={e['severity']} | ai={e['ai']}"
            for e in events
        )
        system = (
            "Create a compact doctor-facing session summary.\n"
            "Include: severity trend, main symptoms and triggers, functional impact, "
            "risk notes, advice given, and next-step recommendation."
        )
        summary_text = _chat(system, f"Session ID: {session_id}\n\nData:\n{history_blob}")

        messages = (
            [{"role": "user",      "message": e["user"], "severity": e["severity"], "timestamp": e["timestamp"]} for e in events]
          + [{"role": "assistant", "message": e["ai"],   "severity": e["severity"], "timestamp": e["timestamp"]} for e in events]
        )
        return {
            "session_id":   session_id,
            "generated_at": _now_iso(),
            "summary_text": summary_text,
            "event_count":  len(events),
            "messages":     messages,
        }


# ── App startup / singleton ───────────────────────────────────────────────────

service: Optional[OCDRAGService] = None

@app.on_event("startup")
def startup_event():
    global service
    service = OCDRAGService()
    print("OCDRAGService initialised and ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Kotlin can ping this to confirm the server is alive."""
    return {"status": "ok", "timestamp": _now_iso()}


@app.post("/session/start", response_model=StartSessionResponse)
def start_session():
    """
    Create a new session. Returns a session_id that Kotlin must store
    and send with every subsequent /chat and /summary request.
    """
    sid = service.create_session()
    return StartSessionResponse(session_id=sid, created_at=_now_iso())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a user message and get the AI response.

    - `session_id`      : from /session/start
    - `message`         : patient's text
    - `kotlin_severity` : optional LOW/MILD/HIGH from your on-device classifier
                          (overrides the server-side model severity if provided)
    """
    try:
        event = service.chat(
            session_id=req.session_id,
            user_input=req.message,
            kotlin_severity=req.kotlin_severity,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        session_id=event["session_id"],
        user_message=event["user"],
        ai_response=event["ai"],
        severity_used=event["severity"],
        severity_model=event["severity_model"],
        severity_kotlin=event["severity_kotlin"] or None,
        timestamp=event["timestamp"],
    )


@app.post("/summary", response_model=SummaryResponse)
def get_summary(req: SummaryRequest):
    """
    Generate a doctor-facing summary for the given session.
    Call this at session end (or on-demand from the doctor app).
    """
    try:
        result = service.summary_for_doctor(req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SummaryResponse(
        session_id=result["session_id"],
        generated_at=result["generated_at"],
        summary_text=result["summary_text"],
        event_count=result["event_count"],
        messages=[MessageItem(**m) for m in result["messages"]],
    )


# ── Entry point ───────────────────────────────────────────────────────────────
# Run with:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)