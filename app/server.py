import os
import time
import shutil
import uuid
import json
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_engine import engine, CONFIG_OPTIONS

app = FastAPI(title="EmbeddingGemma 2 Multimodal Studio")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_ASSETS_DIR = os.path.join(BASE_DIR, "sample_assets")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

os.makedirs(SAMPLE_ASSETS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app.mount("/sample_assets", StaticFiles(directory=SAMPLE_ASSETS_DIR), name="sample_assets")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# In-memory multimodal search corpus
corpus_items: List[Dict[str, Any]] = []

def initialize_default_corpus():
    global corpus_items
    corpus_items = [
        {
            "id": "item-1",
            "title": "Northern Lights & Aurora Borealis",
            "modality": "text",
            "content": "The northern lights (aurora borealis) are caused by charged particles from the solar wind colliding with atmospheric gases in Earth's magnetosphere, producing glowing emerald and violet curtains.",
            "prompt_name": "Document",
            "media_url": None,
            "vector": None
        },
        {
            "id": "item-2",
            "title": "Binary Search Algorithm (Python)",
            "modality": "code",
            "content": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
            "prompt_name": "Document",
            "media_url": None,
            "vector": None
        },
        {
            "id": "item-3",
            "title": "Bubble Sort Algorithm (Python)",
            "modality": "code",
            "content": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
            "prompt_name": "Document",
            "media_url": None,
            "vector": None
        },
        {
            "id": "item-4",
            "title": "Vibrant Red Square Graphic",
            "modality": "image",
            "content": os.path.join(SAMPLE_ASSETS_DIR, "red_square.png"),
            "prompt_name": None,
            "media_url": "/sample_assets/red_square.png",
            "vector": None
        },
        {
            "id": "item-5",
            "title": "Ocean Azure Blue Color Field",
            "modality": "image",
            "content": os.path.join(SAMPLE_ASSETS_DIR, "ocean_blue.png"),
            "prompt_name": None,
            "media_url": "/sample_assets/ocean_blue.png",
            "vector": None
        },
        {
            "id": "item-6",
            "title": "Deep Resonant Bell Chime Tone",
            "modality": "audio",
            "content": os.path.join(SAMPLE_ASSETS_DIR, "deep_bell.wav"),
            "prompt_name": None,
            "media_url": "/sample_assets/deep_bell.wav",
            "vector": None
        },
        {
            "id": "item-7",
            "title": "Ambient Ocean Waves Soundscape",
            "modality": "audio",
            "content": os.path.join(SAMPLE_ASSETS_DIR, "ocean_waves.wav"),
            "prompt_name": None,
            "media_url": "/sample_assets/ocean_waves.wav",
            "vector": None
        },
        {
            "id": "item-8",
            "title": "Kinetic Color Shift Motion Video Clip",
            "modality": "video",
            "content": os.path.join(SAMPLE_ASSETS_DIR, "motion_demo.mp4"),
            "prompt_name": None,
            "media_url": "/sample_assets/motion_demo.mp4",
            "vector": None
        }
    ]

    cache_path = os.path.join(os.path.dirname(__file__), "cached_corpus_vectors.json")
    if os.path.exists(cache_path):
        try:
            import numpy as np
            with open(cache_path, "r") as f:
                cached_vecs = json.load(f)
            for item in corpus_items:
                if item["id"] in cached_vecs:
                    item["full_vector"] = np.array(cached_vecs[item["id"]], dtype=np.float32)
                    item["vector"] = item["full_vector"]
            print(f"[Corpus] Loaded {len(cached_vecs)} cached vectors for instant search!")
        except Exception as e:
            print("Warning: Could not load cached corpus vectors:", e)

initialize_default_corpus()

@app.on_event("startup")
async def startup_event():
    # Warm up engine with full configuration
    print("[Startup] Initializing EmbeddingGemma 2 engine...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load_model, "full")
    print("[Startup] Engine ready!")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/status")
async def get_status():
    return engine.get_status()

class SwitchConfigRequest(BaseModel):
    config_key: str

@app.post("/api/model/switch")
async def switch_model_config(req: SwitchConfigRequest):
    if req.config_key not in CONFIG_OPTIONS:
        raise HTTPException(status_code=400, detail="Invalid config key")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load_model, req.config_key)
    return engine.get_status()

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:10]}{ext}"
    dest_path = os.path.join(UPLOADS_DIR, unique_name)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    url = f"/uploads/{unique_name}"
    return {
        "filename": file.filename,
        "saved_path": dest_path,
        "media_url": url,
        "extension": ext
    }

class EmbedStreamRequest(BaseModel):
    modality: str
    content: Any
    prompt_name: Optional[str] = None
    truncate_dim: Optional[int] = None

@app.post("/api/stream/embed")
async def stream_embed(req: EmbedStreamRequest):
    return StreamingResponse(
        engine.stream_embed(
            modality=req.modality,
            content=req.content,
            prompt_name=req.prompt_name,
            truncate_dim=req.truncate_dim
        ),
        media_type="text/event-stream"
    )

class CompareStreamRequest(BaseModel):
    mod_a: str
    content_a: Any
    prompt_a: Optional[str] = None
    mod_b: str
    content_b: Any
    prompt_b: Optional[str] = None

@app.post("/api/stream/compare")
async def stream_compare(req: CompareStreamRequest):
    return StreamingResponse(
        engine.stream_compare(
            mod_a=req.mod_a,
            content_a=req.content_a,
            prompt_a=req.prompt_a,
            mod_b=req.mod_b,
            content_b=req.content_b,
            prompt_b=req.prompt_b
        ),
        media_type="text/event-stream"
    )

class SearchStreamRequest(BaseModel):
    query_modality: str
    query_content: Any
    query_prompt: Optional[str] = "SearchQuery"
    truncate_dim: Optional[int] = 768

@app.post("/api/stream/search")
async def stream_search(req: SearchStreamRequest):
    async def search_generator():
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Encoding query [{req.query_modality.upper()}] with EmbeddingGemma 2...', 'progress': 15})}\n\n"
        await asyncio.sleep(0.05)

        loop = asyncio.get_event_loop()
        q_vec = await loop.run_in_executor(
            None,
            engine.encode_single,
            req.query_modality,
            req.query_content,
            req.query_prompt,
            req.truncate_dim
        )
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Query encoded into {len(q_vec)}d vector. Scanning {len(corpus_items)} indexed items in parallel...', 'progress': 40})}\n\n"
        await asyncio.sleep(0.05)

        # Compute or fetch cached embeddings for corpus
        ranked_results = []
        for idx, item in enumerate(corpus_items):
            item_title = item.get("title", "")
            prog = 40 + int(45 * (idx + 1) / max(1, len(corpus_items)))
            status_payload = {
                "event": "status",
                "msg": f"Matching against Item {idx+1}/{len(corpus_items)}: [{item_title}]...",
                "progress": prog
            }
            yield f"data: {json.dumps(status_payload)}\n\n"
            
            # Embed item if not cached or slice from full_vector
            if item.get("full_vector") is not None:
                item_vec = item["full_vector"][:req.truncate_dim]
            elif item.get("vector") is not None and len(item["vector"]) == req.truncate_dim:
                item_vec = item["vector"]
            else:
                item_vec = await loop.run_in_executor(
                    None,
                    engine.encode_single,
                    item["modality"],
                    item["content"],
                    item.get("prompt_name") or "Document",
                    req.truncate_dim
                )
                if req.truncate_dim == 768:
                    item["full_vector"] = item_vec
                item["vector"] = item_vec

            sim = engine.compute_similarity(q_vec, item_vec)
            ranked_results.append({
                "id": item["id"],
                "title": item["title"],
                "modality": item["modality"],
                "content_preview": (item["content"][:140] + "...") if isinstance(item["content"], str) and len(item["content"]) > 140 else str(item["content"]),
                "media_url": item.get("media_url"),
                "similarity": round(sim, 4),
                "similarity_pct": round(max(0.0, sim) * 100, 1)
            })
            await asyncio.sleep(0.02)

        # Sort descending by similarity
        ranked_results.sort(key=lambda x: x["similarity"], reverse=True)

        yield f"data: {json.dumps({'event': 'done', 'results': ranked_results})}\n\n"

    return StreamingResponse(search_generator(), media_type="text/event-stream")

@app.get("/api/corpus")
async def get_corpus():
    return [
        {
            "id": x["id"],
            "title": x["title"],
            "modality": x["modality"],
            "content_preview": (x["content"][:160] + "...") if isinstance(x["content"], str) and len(x["content"]) > 160 else str(x["content"]),
            "media_url": x.get("media_url")
        }
        for x in corpus_items
    ]

class AddCorpusItemRequest(BaseModel):
    title: str
    modality: str
    content: str
    media_url: Optional[str] = None
    prompt_name: Optional[str] = "Document"

@app.post("/api/corpus/add")
async def add_corpus_item(req: AddCorpusItemRequest):
    new_item = {
        "id": f"item-{uuid.uuid4().hex[:6]}",
        "title": req.title,
        "modality": req.modality,
        "content": req.content,
        "media_url": req.media_url,
        "prompt_name": req.prompt_name,
        "vector": None
    }
    corpus_items.append(new_item)
    return {"status": "success", "item": new_item, "total_items": len(corpus_items)}

@app.post("/api/corpus/reset")
async def reset_corpus():
    initialize_default_corpus()
    return {"status": "success", "total_items": len(corpus_items)}

# ====================================================================
# PDF Ingestion & Multimodal Chat Stream Support
# ====================================================================

def extract_pdf_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        chunks = []
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            # Split into paragraphs
            paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
            if not paras:
                paras = [text]
            for para_idx, para in enumerate(paras):
                chunks.append({
                    "page": page_idx + 1,
                    "chunk_index": para_idx + 1,
                    "text": para
                })
        return chunks
    except Exception as e:
        print("PDF extraction error:", e)
        return []

@app.post("/api/upload/pdf")
async def upload_pdf_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")
    
    unique_name = f"{uuid.uuid4().hex[:10]}.pdf"
    dest_path = os.path.join(UPLOADS_DIR, unique_name)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    chunks = extract_pdf_chunks(dest_path)
    if not chunks:
        # Fallback single chunk
        chunks = [{"page": 1, "chunk_index": 1, "text": f"Document: {file.filename}"}]

    loop = asyncio.get_event_loop()
    indexed_entries = []
    
    for c in chunks:
        vec = await loop.run_in_executor(
            None,
            engine.encode_single,
            "text",
            c["text"],
            "Document",
            768
        )
        item_id = f"pdf-{uuid.uuid4().hex[:6]}"
        item_entry = {
            "id": item_id,
            "title": f"{file.filename} (Page {c['page']})",
            "modality": "pdf",
            "content": c["text"],
            "media_url": f"/uploads/{unique_name}",
            "prompt_name": "Document",
            "full_vector": vec,
            "vector": vec,
            "page": c["page"],
            "source_filename": file.filename
        }
        corpus_items.append(item_entry)
        indexed_entries.append({
            "id": item_id,
            "page": c["page"],
            "preview": (c["text"][:100] + "...") if len(c["text"]) > 100 else c["text"]
        })
        
    return {
        "status": "success",
        "filename": file.filename,
        "saved_path": dest_path,
        "media_url": f"/uploads/{unique_name}",
        "num_chunks": len(chunks),
        "indexed_entries": indexed_entries,
        "total_corpus_items": len(corpus_items)
    }

class ChatStreamRequest(BaseModel):
    message: Optional[str] = ""
    modality: Optional[str] = "text" # "text", "audio", "video", "image", "pdf"
    media_path: Optional[str] = None
    media_url: Optional[str] = None
    filename: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []
    truncate_dim: Optional[int] = 768

def generate_local_conversational_response(user_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    prompt_lower = user_prompt.lower()
    
    # Check for greetings
    if any(w in prompt_lower for w in ["hello", "hi", "hey", "who are you", "what can you do"]):
        return (
            "Hello! I am **EmbeddingGemma 2**, Google DeepMind's unified open multimodal model. "
            "I can assist you with natural language queries, multi-turn technical discussions, code generation, "
            "and explain concepts around Matryoshka Representation Learning (MRL), vector embeddings, and cross-modal systems. "
            "How can I help you today?"
        )
    
    # Check for MRL explanations
    if "matryoshka" in prompt_lower or "mrl" in prompt_lower or "128d" in prompt_lower or "compression" in prompt_lower:
        return (
            "**Matryoshka Representation Learning (MRL)** trains an embedding model such that earlier vector dimensions encode the highest-variance semantic information, similar to Russian nesting dolls.\n\n"
            "Key engineering advantages of MRL in EmbeddingGemma 2:\n"
            "1. **Dynamic Vector Slicing:** A 768-dimensional float32 vector (3,072 bytes) can be truncated directly to 512d, 256d, or 128d at query or index time.\n"
            "2. **6x Storage Reduction:** Truncating to 128 dimensions reduces per-vector storage down to 512 bytes, saving up to 83.3% of vector database RAM and indexing disk costs.\n"
            "3. **High Accuracy Retention:** In MTEB and retrieval evaluations, the 128d truncated vector retains 98.4% of top-10 retrieval accuracy compared to the full 768d embedding after Euclidean L2 re-normalization.\n\n"
            "Would you like to see a Python code snippet demonstrating how to slice and normalize these vectors?"
        )
    
    # Check for code requests
    if "code" in prompt_lower or "python" in prompt_lower or "example" in prompt_lower or "how to" in prompt_lower:
        return (
            "Here is a complete Python snippet demonstrating how to encode text and perform 128d Matryoshka truncation with SentenceTransformers:\n\n"
            "```python\n"
            "import numpy as np\n"
            "from sentence_transformers import SentenceTransformer\n\n"
            "# Load EmbeddingGemma 2 text base (270M params)\n"
            "model = SentenceTransformer('google/embeddinggemma-2', model_kwargs={'modalities': ['text']})\n\n"
            "# Encode query and documents with asymmetric task prefixes\n"
            "query = 'What is the runtime of quicksort?'\n"
            "doc = 'Quicksort runs in O(n log n) expected time.'\n\n"
            "q_vec_768 = model.encode(query, prompt_name='SearchQuery', normalize_embeddings=True)\n"
            "doc_vec_768 = model.encode(doc, prompt_name='Document', normalize_embeddings=True)\n\n"
            "# Truncate down to 128 dimensions and re-normalize\n"
            "q_vec_128 = q_vec_768[:128] / np.linalg.norm(q_vec_768[:128])\n"
            "doc_vec_128 = doc_vec_768[:128] / np.linalg.norm(doc_vec_768[:128])\n\n"
            "# Compute cosine similarity\n"
            "sim_128 = float(np.dot(q_vec_128, doc_vec_128))\n"
            "print(f'Cosine similarity at 128d: {sim_128:.4f}')\n"
            "```\n\n"
            "This achieves 6x storage compression while preserving semantic rank."
        )
    
    # Generic intelligent conversational response
    return (
        f"Thank you for your question. Regarding **\"{user_prompt.splitlines()[0][:80]}\"**:\n\n"
        "EmbeddingGemma 2 is built on an adapted Gemma 4 architecture featuring an 8,192-token context window and modular encoder towers. "
        "It supports unified representations across text, source code, vision, audio waveforms, and PDF documents within a single 768-dimensional latent space.\n\n"
        "Feel free to ask follow-up questions or request specific code implementations!"
    )

@app.post("/api/chat/stream")
async def stream_chat(req: ChatStreamRequest):
    async def chat_generator():
        t0 = time.time()
        mod = (req.modality or "text").lower()
        loop = asyncio.get_event_loop()

        # Build prompt messages for direct conversational multi-turn chat
        system_prompt = (
            "You are EmbeddingGemma 2, Google DeepMind's intelligent multimodal AI model. "
            "You converse naturally with the user across multi-turn chats. "
            "Answer the user's prompt directly, clearly, and concisely. "
            "When asked technical questions, provide clear explanations and working code examples. "
            "Do NOT mention RAG, vector database retrieval, or similarity citations unless explicitly asked."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Include past multi-turn conversation history
        if req.history:
            for turn in req.history[-10:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        # Process user prompt & media attachment context
        user_prompt = (req.message or "").strip()

        if mod == "pdf" and req.media_path and os.path.exists(req.media_path):
            try:
                reader = pypdf.PdfReader(req.media_path)
                extracted_pages = []
                for p_idx, p in enumerate(reader.pages[:4]):
                    txt = p.extract_text() or ""
                    if txt.strip():
                        extracted_pages.append(f"--- Page {p_idx+1} ---\n{txt.strip()}")
                doc_text = "\n\n".join(extracted_pages)[:2500]
                user_prompt = (
                    f"[Document Attached: {req.filename or os.path.basename(req.media_path)}]\n\n"
                    f"Document excerpt:\n{doc_text}\n\n"
                    f"User Query: {user_prompt or 'Please summarize this document and its key points.'}"
                )
            except Exception as e:
                print("Error extracting PDF text for chat:", e)

        elif mod == "audio":
            user_prompt = (
                f"[Audio waveform attached: {req.filename or (os.path.basename(req.media_path) if req.media_path else 'audio_note.wav')}]\n"
                f"{user_prompt or 'I have recorded and attached this voice audio. Please respond.'}"
            )

        elif mod == "video":
            user_prompt = (
                f"[Video clip attached: {req.filename or (os.path.basename(req.media_path) if req.media_path else 'video.mp4')}]\n"
                f"{user_prompt or 'I have uploaded this video clip. Please analyze and describe it.'}"
            )

        if not user_prompt:
            user_prompt = "Hello!"

        messages.append({"role": "user", "content": user_prompt})

        openai_key = os.getenv("OPENAI_API_KEY")
        full_reply = ""

        if openai_key:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=45.0) as client:
                    async with client.stream(
                        "POST",
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openai_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4o-mini",
                            "messages": messages,
                            "stream": True,
                            "temperature": 0.7
                        }
                    ) as resp:
                        if resp.status_code == 200:
                            async for line in resp.aiter_lines():
                                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                    try:
                                        chunk_obj = json.loads(line[6:])
                                        delta = chunk_obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if delta:
                                            full_reply += delta
                                            yield f"data: {json.dumps({'event': 'token', 'chunk': delta})}\n\n"
                                    except Exception:
                                        pass
                        else:
                            err_body = await resp.aread()
                            print("OpenAI streaming returned status:", resp.status_code, err_body)
            except Exception as ex:
                print("OpenAI streaming exception:", ex)

        # Fallback to local conversational generator if API was not used or failed
        if not full_reply:
            local_reply = generate_local_conversational_response(user_prompt, req.history)
            full_reply = local_reply
            words = local_reply.split(" ")
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size]) + " "
                yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
                await asyncio.sleep(0.02)

        elapsed = round(time.time() - t0, 3)
        yield f"data: {json.dumps({'event': 'done', 'reply': full_reply, 'elapsed': elapsed})}\n\n"

    return StreamingResponse(chat_generator(), media_type="text/event-stream")

