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

@app.post("/api/chat/stream")
async def stream_chat(req: ChatStreamRequest):
    async def chat_generator():
        t0 = time.time()
        mod = (req.modality or "text").lower()
        loop = asyncio.get_event_loop()
        
        # 1. Pipeline Status
        if mod == "audio":
            yield f"data: {json.dumps({'event': 'status', 'msg': 'Encoding voice waveform with EmbeddingGemma 2 Audio Tower...', 'progress': 20})}\n\n"
        elif mod == "video":
            yield f"data: {json.dumps({'event': 'status', 'msg': 'Extracting video keyframes via PyAV & projecting to 768d space...', 'progress': 20})}\n\n"
        elif mod == "image":
            yield f"data: {json.dumps({'event': 'status', 'msg': 'Encoding image visual features into unified latent space...', 'progress': 20})}\n\n"
        elif mod == "pdf":
            yield f"data: {json.dumps({'event': 'status', 'msg': 'Parsing PDF document and extracting semantic knowledge...', 'progress': 20})}\n\n"
        else:
            yield f"data: {json.dumps({'event': 'status', 'msg': 'Encoding user query with task: SearchQuery prefix...', 'progress': 20})}\n\n"
        
        await asyncio.sleep(0.04)

        # 2. Encode query according to modality
        q_vec = None
        query_label = req.message or ""
        
        if mod == "audio" and req.media_path and os.path.exists(req.media_path):
            q_vec = await loop.run_in_executor(
                None, engine.encode_single, "audio", req.media_path, None, req.truncate_dim
            )
            query_label = f"Voice / Audio Note ({os.path.basename(req.media_path)})"
        elif mod == "video" and req.media_path and os.path.exists(req.media_path):
            q_vec = await loop.run_in_executor(
                None, engine.encode_single, "video", req.media_path, None, req.truncate_dim
            )
            query_label = f"Video Recording ({os.path.basename(req.media_path)})"
        elif mod == "image" and req.media_path and os.path.exists(req.media_path):
            q_vec = await loop.run_in_executor(
                None, engine.encode_single, "image", req.media_path, None, req.truncate_dim
            )
            query_label = f"Image Attachment ({os.path.basename(req.media_path)})"
        elif mod == "pdf" and req.media_path and os.path.exists(req.media_path):
            # Index PDF if not already indexed
            chunks = extract_pdf_chunks(req.media_path)
            for c in chunks:
                vec = await loop.run_in_executor(
                    None, engine.encode_single, "text", c["text"], "Document", req.truncate_dim
                )
                corpus_items.append({
                    "id": f"pdf-{uuid.uuid4().hex[:6]}",
                    "title": f"{req.filename or 'Document'} (Page {c['page']})",
                    "modality": "pdf",
                    "content": c["text"],
                    "media_url": req.media_url,
                    "prompt_name": "Document",
                    "full_vector": vec,
                    "vector": vec,
                    "page": c["page"]
                })
            query_text = req.message.strip() if req.message and req.message.strip() else f"Summary and analysis of {req.filename or 'uploaded document'}"
            q_vec = await loop.run_in_executor(
                None, engine.encode_single, "text", query_text, "SearchQuery", req.truncate_dim
            )
            query_label = f"PDF Document: {req.filename or 'Uploaded File'}"
        else:
            query_text = req.message.strip() if req.message and req.message.strip() else "Overview of EmbeddingGemma 2 capabilities"
            q_vec = await loop.run_in_executor(
                None, engine.encode_single, "text", query_text, "SearchQuery", req.truncate_dim
            )
            query_label = query_text

        # 3. Retrieval against Corpus
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Scanning {len(corpus_items)} multimodal index entries across text, code, audio, video & PDFs...', 'progress': 55})}\n\n"
        await asyncio.sleep(0.04)

        ranked = []
        for item in corpus_items:
            if item.get("full_vector") is not None:
                item_vec = item["full_vector"][:req.truncate_dim]
            elif item.get("vector") is not None and len(item["vector"]) == req.truncate_dim:
                item_vec = item["vector"]
            else:
                item_vec = await loop.run_in_executor(
                    None,
                    engine.encode_single,
                    item["modality"] if item["modality"] != "pdf" else "text",
                    item["content"],
                    item.get("prompt_name") or "Document",
                    req.truncate_dim
                )
                if req.truncate_dim == 768:
                    item["full_vector"] = item_vec
                item["vector"] = item_vec
            
            sim = engine.compute_similarity(q_vec, item_vec)
            ranked.append({
                "id": item["id"],
                "title": item["title"],
                "modality": item["modality"],
                "content": item["content"],
                "media_url": item.get("media_url"),
                "similarity": round(sim, 4),
                "similarity_pct": round(max(0.0, sim) * 100, 1)
            })

        ranked.sort(key=lambda x: x["similarity"], reverse=True)
        top_sources = ranked[:3]

        yield f"data: {json.dumps({'event': 'retrieval', 'sources': top_sources, 'progress': 75})}\n\n"
        await asyncio.sleep(0.05)

        # 4. Synthesize Coherent Multimodal Response
        yield f"data: {json.dumps({'event': 'status', 'msg': 'Synthesizing multimodal assistant response...', 'progress': 85})}\n\n"
        
        # Build synthesis text
        response_paragraphs = []
        best = top_sources[0] if top_sources else None
        
        if mod == "audio":
            response_paragraphs.append(
                f"I processed your audio input through the dedicated **EmbeddingGemma 2 Audio Tower** (16 kHz mono waveform). "
                f"The audio vector maps into the shared 768d latent space and demonstrated highest semantic affinity with **{best['title']}** (similarity: `{best['similarity']:.4f}`)."
            )
        elif mod == "video":
            response_paragraphs.append(
                f"I extracted keyframes from your video using PyAV and projected them through the SigLIP vision encoder into the 768-dimensional manifold. "
                f"Your video most strongly correlates with **{best['title']}** (similarity: `{best['similarity']:.4f}`)."
            )
        elif mod == "pdf":
            response_paragraphs.append(
                f"I analyzed and indexed your PDF document (**{req.filename or 'Uploaded File'}**). "
                f"The highest matching section is **{best['title']}** with a confidence score of `{best['similarity']:.4f}`."
            )
        else:
            response_paragraphs.append(
                f"Based on your query **\"{query_label}\"**, EmbeddingGemma 2 performed an asymmetric cosine search (`task: SearchQuery` vs `task: Document`) across our multimodal corpus. "
                f"The top semantic match is **{best['title']}** with a score of `{best['similarity']:.4f}`."
            )

        if best:
            preview_clean = best['content'].replace('\n', ' ')[:220]
            response_paragraphs.append(
                f"**Primary Evidence Citation:**\n> \"{preview_clean}...\""
            )

        if len(top_sources) > 1:
            secondary = top_sources[1]
            response_paragraphs.append(
                f"Additionally, relevant secondary context was retrieved from **{secondary['title']}** (`{secondary['modality'].upper()}`, similarity: `{secondary['similarity']:.4f}`)."
            )

        full_reply = "\n\n".join(response_paragraphs)

        # Stream words smoothly
        words = full_reply.split(" ")
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size]) + " "
            yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
            await asyncio.sleep(0.03)

        # Done event
        elapsed = round(time.time() - t0, 3)
        yield f"data: {json.dumps({'event': 'done', 'reply': full_reply, 'sources': top_sources, 'elapsed': elapsed})}\n\n"

    return StreamingResponse(chat_generator(), media_type="text/event-stream")

