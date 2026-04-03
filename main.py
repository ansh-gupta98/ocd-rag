"""
OCD RAG Support API — fully cloud-inference build.
Zero torch / transformers / sentence-transformers on the server.
Both LLM and embeddings go through HuggingFace Inference API (HTTP only).
Final image: ~250-350 MB.
"""
import hashlib
import json
import os
import pickle
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
SEVERITY_LEVELS = {"LOW", "MILD", "HIGH"}
MAX_INPUT_CHARS = 3500
HISTORY_WINDOW  = 15

HF_LLM_REPO_ID = os.getenv("HF_LLM_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")
HF_EMBED_MODEL  = os.getenv("HF_EMBED_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
_hf_token: Optional[str] = None

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OCD RAG Support API",
    description="OCD patient support chatbot backend for Kotlin Retrofit.",
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StartSessionResponse(BaseModel):
    session_id: str
    created_at: str

class ChatRequest(BaseModel):
    session_id:      str           = Field(..., description="UUID from /session/start")
    message:         str           = Field(..., description="User message text")
    kotlin_severity: Optional[str] = Field(None, description="LOW | MILD | HIGH")

class MessageItem(BaseModel):
    role: str;  message: str;  severity: str;  timestamp: str

class ChatResponse(BaseModel):
    session_id: str;      user_message: str;    ai_response: str
    severity_used: str;   severity_model: str;  severity_kotlin: Optional[str]
    timestamp: str

class SummaryRequest(BaseModel):
    session_id: str

class SummaryResponse(BaseModel):
    session_id: str;  generated_at: str;  summary_text: str
    event_count: int; messages: List[MessageItem]


# ── HF Inference API: LLM ─────────────────────────────────────────────────────

def _hf_chat(system: str, user: str) -> str:
    """Call HF Inference API for chat — pure HTTP, no local model."""
    headers = {"Authorization": f"Bearer {_hf_token}", "Content-Type": "application/json"}

    # Try modern chat/completions endpoint first
    try:
        resp = requests.post(
            "https://api-inference.huggingface.co/v1/chat/completions",
            headers=headers,
            json={
                "model": HF_LLM_REPO_ID,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": 512,
                "temperature": 0.4,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    # Fallback: legacy text-generation endpoint
    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_LLM_REPO_ID}",
        headers=headers,
        json={"inputs": prompt,
              "parameters": {"max_new_tokens": 512, "temperature": 0.4, "return_full_text": False}},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data[0].get("generated_text", "").strip()
    return str(data).strip()


# ── HF Inference API: Embeddings ──────────────────────────────────────────────

def _hf_embed(texts: List[str]) -> np.ndarray:
    """
    Call HF Inference API for embeddings.
    Tries the new /models/ endpoint first, falls back to legacy /pipeline/.
    Batches in groups of 32 to stay within payload limits.
    """
    headers = {"Authorization": f"Bearer {_hf_token}", "Content-Type": "application/json"}

    # New endpoint (2024+): https://api-inference.huggingface.co/models/<model>
    # Payload: {"inputs": ["text1", "text2"]}
    primary_url = f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}"
    # Legacy fallback (some models still use this)
    legacy_url  = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}"

    all_vecs: List[np.ndarray] = []

    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        raw   = None

        for url in (primary_url, legacy_url):
            try:
                resp = requests.post(
                    url, headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=60,
                )
                if resp.status_code in (404, 410):
                    continue          # try next URL
                resp.raise_for_status()
                raw = resp.json()
                break
            except requests.exceptions.HTTPError:
                continue

        if raw is None:
            raise RuntimeError(
                f"HF embedding API unavailable for model '{HF_EMBED_MODEL}'. "
                "Check your HUGGINGFACEHUB_API_TOKEN and HF_EMBED_MODEL variable."
            )

        vecs = []
        for item in raw:
            # HF may return list[float] or list[list[float]] (token-level → mean pool)
            if isinstance(item, list) and isinstance(item[0], list):
                arr = np.mean(np.array(item, dtype="float32"), axis=0)
            else:
                arr = np.array(item, dtype="float32")
            norm = np.linalg.norm(arr)
            vecs.append(arr / norm if norm > 0 else arr)
        all_vecs.extend(vecs)

    return np.array(all_vecs, dtype="float32")


# ── Pure-numpy FAISS wrapper ──────────────────────────────────────────────────

class CloudFAISS:
    """
    FAISS index backed by HF API embeddings.
    No torch. No sentence-transformers. Just numpy + faiss-cpu.
    """
    def __init__(self):
        self.index = None
        self.texts: List[str] = []
        self.metas: List[Dict] = []

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        import faiss
        vecs = _hf_embed(texts)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.texts.extend(texts)
        self.metas.extend(metadatas or [{} for _ in texts])

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None) -> List[Document]:
        if self.index is None or not self.texts:
            return []
        vec   = _hf_embed([query])
        fetch = min(k * 4, len(self.texts))
        _, idxs = self.index.search(vec, fetch)
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
    def from_documents(cls, docs: List[Document]) -> "CloudFAISS":
        inst = cls()
        inst.add_texts(
            texts=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
        )
        return inst

    def save_local(self, path: str):
        import faiss
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "payload.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metas": self.metas}, f)

    @classmethod
    def load_local(cls, path: str) -> "CloudFAISS":
        import faiss
        p    = Path(path)
        inst = cls()
        inst.index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "payload.pkl", "rb") as f:
            data = pickle.load(f)
        inst.texts = data["texts"]
        inst.metas = data["metas"]
        return inst


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

def _coerce_severity(raw: str) -> str:
    t = (raw or "").strip().upper()
    if "HIGH" in t: return "HIGH"
    if "MILD" in t: return "MILD"
    return "LOW"

def _knowledge_dir_fingerprint(kd: Path) -> str:
    h = hashlib.sha256()
    if not kd.is_dir(): return ""
    for p in sorted(kd.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".txt", ".md", ".pdf"): continue
        try: stat = p.stat()
        except OSError: continue
        h.update(str(p.relative_to(kd)).encode()); h.update(str(stat.st_mtime_ns).encode()); h.update(str(stat.st_size).encode())
    return h.hexdigest()

def _load_docs(kd: Path) -> List[Document]:
    docs: List[Document] = []
    if not kd.is_dir(): return docs
    for f in kd.rglob("*.txt"):  docs.extend(TextLoader(str(f), encoding="utf-8").load())
    for f in kd.rglob("*.md"):   docs.extend(TextLoader(str(f), encoding="utf-8").load())
    for f in kd.rglob("*.pdf"):  docs.extend(PyPDFLoader(str(f)).load())
    return docs

def _build_or_load_db(kd: Path, vs: Path) -> CloudFAISS:
    raw = _load_docs(kd)
    if not raw:
        raise ValueError(f"No docs found under {kd}. Add .txt/.md/.pdf files.")
    splitter   = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    split_docs = splitter.split_documents(raw)
    rebuild    = os.getenv("OCD_REBUILD_VECTOR", "").lower() in ("1", "true", "yes")
    meta_path  = vs / "rag_meta.json"
    src_fp     = _knowledge_dir_fingerprint(kd)

    if not rebuild and (vs / "index.faiss").exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("sources_fingerprint") == src_fp:
                return CloudFAISS.load_local(str(vs))
        except Exception:
            pass

    db = CloudFAISS.from_documents(split_docs)
    db.save_local(str(vs))
    meta_path.write_text(json.dumps({"sources_fingerprint": src_fp, "chunk_count": len(split_docs)}, indent=2))
    return db

def _policy(severity: str) -> str:
    if severity == "LOW":
        return "Provide coping advice and practical self-help. Encourage optional therapist check-in. Console the patient, encourage small talks with friends/family."
    if severity == "MILD":
        return "Provide short coping suggestions, encourage a licensed mental health professional soon — no pressure. Avoid presenting self-help as sufficient."
    return "Keep calm and supportive. Strongly advise urgent contact with a licensed mental health professional. Indian crisis lines: iCall 9152987821, Vandrevala 1860-2662-345."


# ── Service ───────────────────────────────────────────────────────────────────

class OCDRAGService:
    def __init__(self):
        global _hf_token
        _hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not _hf_token:
            raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN in Railway Variables.")

        root = Path(__file__).resolve().parent
        kd   = Path(os.getenv("OCD_KNOWLEDGE_DIR")    or root / "ocd_documentation")
        vs   = Path(os.getenv("OCD_VECTOR_STORE_DIR") or root / "ocd_documentation_vector")
        self.knowledge_db = _build_or_load_db(kd, vs)
        self.history_db   = CloudFAISS()
        self.sessions: Dict[str, List[Dict]] = {}

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = []
        return sid

    def classify_severity(self, text: str) -> str:
        system = (
            "You are a strict mental health triage classifier for OCD support.\n"
            "Classify the severity as exactly one of: LOW, MILD, or HIGH.\n"
            "LOW=minor intrusive thoughts, little impact. "
            "MILD=distress present, some impact. "
            "HIGH=severe distress, impairment, safety risk.\n"
            "Return EXACTLY one token: LOW or MILD or HIGH. No other text."
        )
        return _coerce_severity(_hf_chat(system, text))

    def _history(self, session_id: str, user_query: str) -> str:
        events = self.sessions.get(session_id, [])
        if not events: return "No prior turns."
        lines = []
        for e in events[-HISTORY_WINDOW:]:
            lines.append(f"user: {e['user']}")
            lines.append(f"assistant: {e['ai']}")
        hits = self.history_db.similarity_search(
            f"{session_id} | {user_query}", k=5, filter={"session_id": session_id}
        )
        text = "\n".join(lines)
        if hits:
            text += "\n\nRelevant older memory:\n" + "\n".join(d.page_content for d in hits)
        return text

    def chat(self, session_id: str, user_input: str, kotlin_severity: Optional[str] = None) -> Dict:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Call /session/start first.")
        user_input     = user_input.strip()[:MAX_INPUT_CHARS]
        model_sev      = self.classify_severity(user_input)
        final_sev      = _coerce_severity(kotlin_severity) if kotlin_severity else model_sev
        ctx_docs       = self.knowledge_db.similarity_search(user_input, k=4)
        context        = "\n".join(d.page_content for d in ctx_docs)
        history        = self._history(session_id, user_input)

        system = (
            "You are an OCD support assistant.\n"
            "Use the Clinical Context to answer when possible.\n\n"
            f"Clinical Context:\n{context}\n\n"
            f"Chat History:\n{history}\n\n"
            f"Severity: {final_sev}\nPolicy: {_policy(final_sev)}\n\n"
            "Rules: be empathetic, no diagnosis, no medication, max 150 words."
        )
        ai_text = _hf_chat(system, user_input)
        event = {
            "timestamp": _now_iso(), "session_id": session_id,
            "user": user_input, "ai": ai_text,
            "severity": final_sev, "severity_model": model_sev,
            "severity_kotlin": (kotlin_severity or "").upper(),
        }
        self.sessions[session_id].append(event)
        if self.history_db.index is not None or len(self.sessions[session_id]) > 0:
            self.history_db.add_texts(
                texts=[f"{event['timestamp']} | user: {event['user']}",
                       f"{event['timestamp']} | assistant: {event['ai']}"],
                metadatas=[{"session_id": session_id, "role": "user"},
                           {"session_id": session_id, "role": "assistant"}],
            )
        return event

    def summary_for_doctor(self, session_id: str) -> Dict:
        events = self.sessions.get(session_id, [])
        if not events:
            return {"session_id": session_id, "generated_at": _now_iso(),
                    "summary_text": "No conversation history.", "event_count": 0, "messages": []}
        blob = "\n".join(
            f"{e['timestamp']} | user={e['user']} | severity={e['severity']} | ai={e['ai']}"
            for e in events
        )
        system = (
            "Create a compact doctor-facing session summary. "
            "Include: severity trend, symptoms/triggers, functional impact, "
            "risk notes, advice given, next-step recommendation."
        )
        summary = _hf_chat(system, f"Session: {session_id}\n\n{blob}")
        messages = (
            [{"role": "user",      "message": e["user"], "severity": e["severity"], "timestamp": e["timestamp"]} for e in events]
          + [{"role": "assistant", "message": e["ai"],   "severity": e["severity"], "timestamp": e["timestamp"]} for e in events]
        )
        return {"session_id": session_id, "generated_at": _now_iso(),
                "summary_text": summary, "event_count": len(events), "messages": messages}


# ── Startup ───────────────────────────────────────────────────────────────────

service: Optional[OCDRAGService] = None

@app.on_event("startup")
def startup_event():
    global service
    service = OCDRAGService()
    print("✅ OCDRAGService ready (cloud-inference mode, no torch).")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": _now_iso()}

@app.post("/session/start", response_model=StartSessionResponse)
def start_session():
    return StartSessionResponse(session_id=service.create_session(), created_at=_now_iso())

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        e = service.chat(req.session_id, req.message, req.kotlin_severity)
    except KeyError as ex:
        raise HTTPException(404, detail=str(ex))
    except Exception as ex:
        raise HTTPException(500, detail=str(ex))
    return ChatResponse(
        session_id=e["session_id"], user_message=e["user"], ai_response=e["ai"],
        severity_used=e["severity"], severity_model=e["severity_model"],
        severity_kotlin=e["severity_kotlin"] or None, timestamp=e["timestamp"],
    )

@app.post("/summary", response_model=SummaryResponse)
def get_summary(req: SummaryRequest):
    try:
        r = service.summary_for_doctor(req.session_id)
    except Exception as ex:
        raise HTTPException(500, detail=str(ex))
    return SummaryResponse(
        session_id=r["session_id"], generated_at=r["generated_at"],
        summary_text=r["summary_text"], event_count=r["event_count"],
        messages=[MessageItem(**m) for m in r["messages"]],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)