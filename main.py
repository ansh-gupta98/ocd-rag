import hashlib
import json
import os
import uuid
import requests
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
SEVERITY_LEVELS = {"LOW", "MILD", "HIGH"}
MAX_INPUT_CHARS = 3500
HISTORY_WINDOW  = 15

# Ungated, no sentencepiece, works great on HF Inference API
HF_LLM_REPO_ID = os.getenv("HF_LLM_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")
HF_EMBED_MODEL  = os.getenv("HF_EMBED_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
HF_API_URL      = f"https://api-inference.huggingface.co/models/{HF_LLM_REPO_ID}"

_embed_model: Optional[SentenceTransformer] = None
_hf_token:    Optional[str]                 = None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="OCD RAG Support API",
    description="Backend for OCD patient support chatbot. Consumed by Kotlin Retrofit.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StartSessionResponse(BaseModel):
    session_id: str
    created_at: str

class ChatRequest(BaseModel):
    session_id:      str           = Field(..., description="UUID from /session/start")
    message:         str           = Field(..., description="User message text")
    kotlin_severity: Optional[str] = Field(None, description="LOW | MILD | HIGH from Kotlin classifier")

class MessageItem(BaseModel):
    role:      str
    message:   str
    severity:  str
    timestamp: str

class ChatResponse(BaseModel):
    session_id:      str
    user_message:    str
    ai_response:     str
    severity_used:   str
    severity_model:  str
    severity_kotlin: Optional[str]
    timestamp:       str

class SummaryRequest(BaseModel):
    session_id: str

class SummaryResponse(BaseModel):
    session_id:   str
    generated_at: str
    summary_text: str
    event_count:  int
    messages:     List[MessageItem]


# ── Lightweight HF Inference API caller (replaces ChatHuggingFace + torch) ───

def _hf_chat(system: str, user: str) -> str:
    """
    Calls the HuggingFace Inference API directly using chat/completions format.
    No torch, no transformers, no local model — just HTTP.
    """
    headers = {
        "Authorization": f"Bearer {_hf_token}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": HF_LLM_REPO_ID,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        "max_tokens":   512,
        "temperature":  0.4,
        "stream":       False,
    }
    # Try chat/completions endpoint first (newer models)
    chat_url = f"https://api-inference.huggingface.co/v1/chat/completions"
    try:
        resp = requests.post(chat_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    # Fallback: legacy text-generation endpoint
    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    legacy_payload = {
        "inputs":      prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.4, "return_full_text": False},
    }
    resp = requests.post(HF_API_URL, headers=headers, json=legacy_payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data[0].get("generated_text", "").strip()
    return str(data).strip()


# ── Minimal FAISS wrapper using sentence-transformers directly ────────────────
# Avoids langchain_huggingface (which pulls in transformers + torch)

class LightFAISS:
    """
    Thin wrapper: sentence-transformers for embeddings, faiss-cpu for search.
    Replaces HuggingFaceEmbeddings from langchain_huggingface entirely.
    """
    def __init__(self, model: SentenceTransformer):
        self.model   = model
        self.texts:  List[str]       = []
        self.metas:  List[Dict]      = []
        self.index   = None          # faiss.IndexFlatL2

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype="float32")

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        import faiss
        vecs = self._embed(texts)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])  # inner product = cosine on normalized
        self.index.add(vecs)
        self.texts.extend(texts)
        self.metas.extend(metadatas or [{} for _ in texts])

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None) -> List[Document]:
        if self.index is None or not self.texts:
            return []
        import faiss
        vec = self._embed([query])
        k_actual = min(k * 3, len(self.texts))  # over-fetch for filter
        scores, idxs = self.index.search(vec, k_actual)
        results = []
        for idx in idxs[0]:
            if idx < 0 or idx >= len(self.texts):
                continue
            meta = self.metas[idx]
            if filter and not all(meta.get(fk) == fv for fk, fv in filter.items()):
                continue
            results.append(Document(page_content=self.texts[idx], metadata=meta))
            if len(results) >= k:
                break
        return results

    @classmethod
    def from_documents(cls, docs: List[Document], model: SentenceTransformer) -> "LightFAISS":
        inst = cls(model)
        inst.add_texts(
            texts=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
        )
        return inst

    def save_local(self, path: str):
        import faiss, pickle
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "texts_metas.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metas": self.metas}, f)

    @classmethod
    def load_local(cls, path: str, model: SentenceTransformer) -> "LightFAISS":
        import faiss, pickle
        p = Path(path)
        inst = cls(model)
        inst.index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "texts_metas.pkl", "rb") as f:
            data = pickle.load(f)
        inst.texts = data["texts"]
        inst.metas = data["metas"]
        return inst


# ── Client init ───────────────────────────────────────────────────────────────

def _init_clients() -> None:
    global _embed_model, _hf_token
    if _embed_model is not None:
        return

    _hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not _hf_token:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in Railway Variables.")

    # ~90MB download, CPU-only, no torch GPU overhead
    _embed_model = SentenceTransformer(HF_EMBED_MODEL, device="cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

def _coerce_severity(raw: str) -> str:
    text = (raw or "").strip().upper()
    if "HIGH" in text: return "HIGH"
    if "MILD" in text: return "MILD"
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
        print(f"Warning: {knowledge_dir} does not exist.")
        return docs
    for f in knowledge_dir.rglob("*.txt"):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    for f in knowledge_dir.rglob("*.md"):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    for f in knowledge_dir.rglob("*.pdf"):
        print(f"Loading PDF {f}...")
        docs.extend(PyPDFLoader(str(f)).load())
    return docs

def _build_or_load_knowledge_db(knowledge_dir: Path, vector_dir: Path) -> LightFAISS:
    raw_docs = _load_documents_from_directory(knowledge_dir)
    if not raw_docs:
        raise ValueError(f"No docs found under {knowledge_dir}. Add .txt/.md/.pdf files.")

    splitter   = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    split_docs = splitter.split_documents(raw_docs)

    rebuild    = os.getenv("OCD_REBUILD_VECTOR", "").lower() in ("1", "true", "yes")
    meta_path  = vector_dir / "rag_meta.json"
    index_file = vector_dir / "index.faiss"
    src_fp     = _knowledge_dir_fingerprint(knowledge_dir)

    if not rebuild and index_file.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("sources_fingerprint") == src_fp:
                return LightFAISS.load_local(str(vector_dir), _embed_model)
        except Exception:
            pass

    db = LightFAISS.from_documents(split_docs, _embed_model)
    db.save_local(str(vector_dir))
    meta_path.write_text(
        json.dumps({"sources_fingerprint": src_fp, "chunk_count": len(split_docs)}, indent=2),
        encoding="utf-8",
    )
    return db

def _policy_for_severity(severity: str) -> str:
    if severity == "LOW":
        return (
            "Provide coping advice and practical self-help. "
            "Encourage optional check-in with a therapist if symptoms persist. "
            "Console the patient, encourage small talks with friends/family."
        )
    if severity == "MILD":
        return (
            "Provide short coping suggestions but encourage meeting a licensed mental health "
            "professional soon — no pressure. Avoid presenting self-help as sufficient."
        )
    return (
        "Keep calm and supportive. Strongly advise urgent contact with a licensed mental health "
        "professional. If self-harm risk present, advise emergency services. "
        "Indian crisis lines: iCall 9152987821, Vandrevala 1860-2662-345."
    )


# ── Service ───────────────────────────────────────────────────────────────────

class OCDRAGService:
    def __init__(self) -> None:
        _init_clients()
        root = Path(__file__).resolve().parent
        kd   = Path(os.getenv("OCD_KNOWLEDGE_DIR")    or root / "ocd_documentation")
        vs   = Path(os.getenv("OCD_VECTOR_STORE_DIR") or root / "ocd_documentation_vector")
        self.knowledge_db = _build_or_load_knowledge_db(kd, vs)
        self.history_db   = LightFAISS(_embed_model)
        self.history_db.add_texts(["bootstrap memory"], [{"session_id": "__init__"}])
        self.sessions: Dict[str, List[Dict]] = {}

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = []
        return sid

    def classify_severity(self, user_input: str) -> str:
        system = (
            "You are a strict mental health triage classifier for OCD support.\n"
            "Classify the user message severity as exactly one of: LOW, MILD, or HIGH.\n"
            "LOW  – minor intrusive thoughts, little functional impact.\n"
            "MILD – distress present, some functional impact, can still manage.\n"
            "HIGH – severe distress, impairment, safety risk, panic, inability to function.\n"
            "Return EXACTLY one token: LOW or MILD or HIGH. No other text."
        )
        return _coerce_severity(_hf_chat(system, user_input))

    def _render_recent_history(self, session_id: str, user_query: str) -> str:
        events = self.sessions.get(session_id, [])
        if not events:
            return "No prior turns."
        lines = []
        for e in events[-HISTORY_WINDOW:]:
            lines.append(f"user: {e['user']}")
            lines.append(f"assistant: {e['ai']}")
        memory_hits  = self.history_db.similarity_search(
            f"{session_id} | {user_query}", k=5, filter={"session_id": session_id}
        )
        history_text = "\n".join(lines)
        if memory_hits:
            history_text += "\n\nRelevant older memory:\n" + "\n".join(
                d.page_content for d in memory_hits
            )
        return history_text

    def chat(self, session_id: str, user_input: str, kotlin_severity: Optional[str] = None) -> Dict:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Call /session/start first.")

        user_input     = user_input.strip()[:MAX_INPUT_CHARS]
        model_severity = self.classify_severity(user_input)
        final_severity = _coerce_severity(kotlin_severity) if kotlin_severity else model_severity

        context_docs = self.knowledge_db.similarity_search(user_input, k=4)
        context      = "\n".join(doc.page_content for doc in context_docs)
        history_text = self._render_recent_history(session_id, user_query=user_input)

        system = (
            "You are an OCD support assistant.\n"
            "You MUST use the provided Clinical Context to answer.\n\n"
            f"Clinical Context:\n{context}\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Severity: {final_severity}\n"
            f"Policy: {_policy_for_severity(final_severity)}\n\n"
            "Rules:\n"
            "- Refer to context when possible\n"
            "- Be empathetic\n"
            "- No diagnosis or medication instructions\n"
            "- Max 150 words"
        )
        ai_text = _hf_chat(system, user_input)

        event = {
            "timestamp":       _now_iso(),
            "session_id":      session_id,
            "user":            user_input,
            "ai":              ai_text,
            "severity":        final_severity,
            "severity_model":  model_severity,
            "severity_kotlin": (kotlin_severity or "").upper(),
        }
        self.sessions[session_id].append(event)
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

    def summary_for_doctor(self, session_id: str) -> Dict:
        events = self.sessions.get(session_id, [])
        if not events:
            return {
                "session_id": session_id, "generated_at": _now_iso(),
                "summary_text": "No conversation history available.",
                "event_count": 0, "messages": [],
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
        summary_text = _hf_chat(system, f"Session ID: {session_id}\n\nData:\n{history_blob}")
        messages = (
            [{"role": "user",      "message": e["user"], "severity": e["severity"], "timestamp": e["timestamp"]} for e in events]
          + [{"role": "assistant", "message": e["ai"],   "severity": e["severity"], "timestamp": e["timestamp"]} for e in events]
        )
        return {
            "session_id": session_id, "generated_at": _now_iso(),
            "summary_text": summary_text, "event_count": len(events), "messages": messages,
        }


# ── App lifecycle ─────────────────────────────────────────────────────────────

service: Optional[OCDRAGService] = None

@app.on_event("startup")
def startup_event():
    global service
    service = OCDRAGService()
    print("✅ OCDRAGService ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": _now_iso()}

@app.post("/session/start", response_model=StartSessionResponse)
def start_session():
    sid = service.create_session()
    return StartSessionResponse(session_id=sid, created_at=_now_iso())

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)