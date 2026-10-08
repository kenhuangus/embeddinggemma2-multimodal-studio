import os
import time
import json
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, List
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers.video_utils import load_video

MODEL_ID = "google/embeddinggemma-2"

CONFIG_OPTIONS = {
    "full": {
        "name": "Full Multimodal",
        "description": "Text, Code, Vision (Images/Video), and Audio (740M params)",
        "config_kwargs": {},
        "params": "740M",
        "supported_modalities": ["text", "code", "image", "audio", "video", "interleaved"]
    },
    "text_vision": {
        "name": "Text + Vision",
        "description": "Text, Code, Images, and Video frames (440M params)",
        "config_kwargs": {"audio_config": None},
        "params": "440M",
        "supported_modalities": ["text", "code", "image", "video"]
    },
    "text_audio": {
        "name": "Text + Audio",
        "description": "Text, Code, and Speech/Audio (570M params)",
        "config_kwargs": {"vision_config": None},
        "params": "570M",
        "supported_modalities": ["text", "code", "audio"]
    },
    "text_only": {
        "name": "Text & Code Only",
        "description": "Lightweight base for text and codebase search (270M params)",
        "config_kwargs": {"vision_config": None, "audio_config": None},
        "params": "270M",
        "supported_modalities": ["text", "code"]
    }
}

class EmbeddingGemmaEngine:
    def __init__(self):
        self.current_config = "full"
        self.model: Optional[SentenceTransformer] = None
        self.is_loading = False
        self.load_lock = asyncio.Lock()
        self.load_time_seconds = 0.0

    def get_status(self) -> Dict[str, Any]:
        cfg_info = CONFIG_OPTIONS[self.current_config]
        return {
            "model_id": MODEL_ID,
            "is_loaded": self.model is not None,
            "is_loading": self.is_loading,
            "current_config": self.current_config,
            "config_name": cfg_info["name"],
            "parameters": cfg_info["params"],
            "description": cfg_info["description"],
            "supported_modalities": cfg_info["supported_modalities"],
            "available_configs": {k: {"name": v["name"], "params": v["params"], "description": v["description"]} for k, v in CONFIG_OPTIONS.items()},
            "load_time": f"{self.load_time_seconds:.2f}s",
            "device": "CPU" if not torch.cuda.is_available() else "CUDA"
        }

    def load_model(self, config_key: str = "full"):
        if config_key not in CONFIG_OPTIONS:
            config_key = "full"
        print(f"[Engine] Loading configuration '{config_key}'...")
        start = time.time()
        self.is_loading = True
        try:
            cfg = CONFIG_OPTIONS[config_key]
            self.model = SentenceTransformer(
                MODEL_ID,
                config_kwargs=cfg["config_kwargs"]
            )
            self.current_config = config_key
            self.load_time_seconds = time.time() - start
            print(f"[Engine] Loaded {cfg['name']} ({cfg['params']}) in {self.load_time_seconds:.2f}s")
        finally:
            self.is_loading = False

    def ensure_model(self):
        if self.model is None and not self.is_loading:
            self.load_model("full")

    def encode_single(
        self,
        modality: str,
        content: Any,
        prompt_name: Optional[str] = None,
        truncate_dim: Optional[int] = None
    ) -> np.ndarray:
        self.ensure_model()
        assert self.model is not None, "Model not loaded"

        modality = modality.lower().strip()
        encode_kwargs: Dict[str, Any] = {
            "normalize_embeddings": True
        }
        if truncate_dim:
            encode_kwargs["truncate_dim"] = truncate_dim

        if modality in ["text", "code"]:
            if prompt_name:
                encode_kwargs["prompt_name"] = prompt_name
            elif modality == "code":
                encode_kwargs["prompt_name"] = "CodeRetrieval"
            else:
                encode_kwargs["prompt_name"] = "SearchQuery"

            text_str = str(content)
            vec = self.model.encode(text_str, **encode_kwargs)
            return np.array(vec, dtype=np.float32)

        elif modality == "image":
            # Image can be path or PIL Image
            if isinstance(content, str):
                vec = self.model.encode({"image": content}, **encode_kwargs)
            elif isinstance(content, Image.Image):
                vec = self.model.encode({"image": content}, **encode_kwargs)
            else:
                raise ValueError(f"Unsupported image type: {type(content)}")
            return np.array(vec, dtype=np.float32)

        elif modality == "audio":
            if not isinstance(content, str) or not os.path.exists(content):
                raise ValueError(f"Audio file path does not exist: {content}")
            vec = self.model.encode({"audio": content}, **encode_kwargs)
            return np.array(vec, dtype=np.float32)

        elif modality == "video":
            if not isinstance(content, str) or not os.path.exists(content):
                raise ValueError(f"Video file path does not exist: {content}")
            # Safely sample 2 frames uniformly using PyAV to maintain low latency and prevent decoder errors
            res = load_video(content, num_frames=2, backend="pyav")
            frames = res[0] if isinstance(res, tuple) else res
            vec = self.model.encode({"video": frames}, **encode_kwargs)
            return np.array(vec, dtype=np.float32)

        elif modality == "interleaved":
            # content is dict with keys: text, and optional image, video, audio paths
            interleaved_dict: Dict[str, Any] = {}
            if "text" in content:
                interleaved_dict["text"] = content["text"]
            if "image" in content and content["image"] and os.path.exists(content["image"]):
                interleaved_dict["image"] = content["image"]
            if "audio" in content and content["audio"] and os.path.exists(content["audio"]):
                interleaved_dict["audio"] = content["audio"]
            if "video" in content and content["video"] and os.path.exists(content["video"]):
                res = load_video(content["video"], num_frames=2, backend="pyav")
                interleaved_dict["video"] = res[0] if isinstance(res, tuple) else res

            vec = self.model.encode(interleaved_dict, **encode_kwargs)
            return np.array(vec, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported modality: '{modality}'")

    def compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        # If both are normalized unit vectors, cosine similarity is dot product
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0: v1 = v1 / norm1
        if norm2 > 0: v2 = v2 / norm2
        return float(np.dot(v1, v2))

    def compute_mrl_spectrum(self, v1_full: np.ndarray, v2_full: np.ndarray) -> List[Dict[str, Any]]:
        dims = [768, 512, 256, 128]
        baseline_sim = None
        results = []
        for d in dims:
            sub1 = v1_full[:d]
            sub2 = v2_full[:d]
            sim = self.compute_similarity(sub1, sub2)
            if baseline_sim is None:
                baseline_sim = sim
            ratio = (sim / baseline_sim * 100.0) if baseline_sim and abs(baseline_sim) > 1e-5 else 100.0
            results.append({
                "dim": d,
                "similarity": round(sim, 4),
                "storage_reduction": f"{768 // d}x",
                "retention_pct": round(min(100.0, max(0.0, ratio)), 1)
            })
        return results

    async def stream_embed(
        self,
        modality: str,
        content: Any,
        prompt_name: Optional[str] = None,
        truncate_dim: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        t0 = time.time()
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Initiating {modality.upper()} embedding pipeline...', 'progress': 10})}\n\n"
        await asyncio.sleep(0.05)

        # Modality preparation
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Preprocessing and validating {modality.upper()} input...', 'progress': 25})}\n\n"
        await asyncio.sleep(0.05)

        if modality == "video":
            yield f"data: {json.dumps({'event': 'status', 'msg': 'Extracting 1 fps video keyframes using PyAV container...', 'progress': 40})}\n\n"
        elif modality == "audio":
            yield f"data: {json.dumps({'event': 'status', 'msg': 'Resampling audio waveform to 16 kHz mono format...', 'progress': 40})}\n\n"
        elif modality in ["text", "code"]:
            prompt_used = prompt_name or ("CodeRetrieval" if modality == "code" else "SearchQuery")
            yield f"data: {json.dumps({'event': 'status', 'msg': f'Applying task prompt instruction prefix: [{prompt_used}]...', 'progress': 40})}\n\n"

        await asyncio.sleep(0.05)
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Executing EmbeddingGemma 2 forward pass (backbone: Gemma 4, dim=768)...', 'progress': 60})}\n\n"
        await asyncio.sleep(0.05)

        # Run encoding in worker thread to prevent event loop blocking
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None,
            self.encode_single,
            modality,
            content,
            prompt_name,
            truncate_dim
        )

        dim = len(vec)
        norm = float(np.linalg.norm(vec))
        preview = [round(float(x), 4) for x in vec[:16]]

        yield f"data: {json.dumps({'event': 'status', 'msg': f'Successfully generated unit-normalized vector with dim={dim} in {time.time()-t0:.2f}s!', 'progress': 90})}\n\n"
        await asyncio.sleep(0.05)

        done_payload = {
            "event": "done",
            "modality": modality,
            "dim": dim,
            "norm": round(norm, 4),
            "preview_slice": preview,
            "elapsed_seconds": round(time.time() - t0, 3),
            "vector": [round(float(x), 5) for x in vec.tolist()]
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    async def stream_compare(
        self,
        mod_a: str,
        content_a: Any,
        prompt_a: Optional[str],
        mod_b: str,
        content_b: Any,
        prompt_b: Optional[str]
    ) -> AsyncGenerator[str, None]:
        t0 = time.time()
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Comparing Input A [{mod_a.upper()}] with Input B [{mod_b.upper()}]...', 'progress': 10})}\n\n"
        await asyncio.sleep(0.05)

        # Embed Input A
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Step 1/3: Encoding Input A ({mod_a.upper()})...', 'progress': 25})}\n\n"
        loop = asyncio.get_event_loop()
        vec_a = await loop.run_in_executor(None, self.encode_single, mod_a, content_a, prompt_a, None)
        yield f"data: {json.dumps({'event': 'vector_a_ready', 'dim_a': len(vec_a), 'progress': 45})}\n\n"
        await asyncio.sleep(0.05)

        # Embed Input B
        yield f"data: {json.dumps({'event': 'status', 'msg': f'Step 2/3: Encoding Input B ({mod_b.upper()})...', 'progress': 60})}\n\n"
        vec_b = await loop.run_in_executor(None, self.encode_single, mod_b, content_b, prompt_b, None)
        yield f"data: {json.dumps({'event': 'vector_b_ready', 'dim_b': len(vec_b), 'progress': 80})}\n\n"
        await asyncio.sleep(0.05)

        # Calculate cosine similarity and MRL spectrum
        yield f"data: {json.dumps({'event': 'status', 'msg': 'Step 3/3: Computing Matryoshka (MRL) dimension spectrum across 768d, 512d, 256d, 128d...', 'progress': 90})}\n\n"
        mrl_table = self.compute_mrl_spectrum(vec_a, vec_b)
        base_sim = self.compute_similarity(vec_a, vec_b)

        result_payload = {
            "event": "done",
            "similarity": round(base_sim, 4),
            "similarity_percent": round(max(0.0, base_sim) * 100, 1),
            "mrl_spectrum": mrl_table,
            "elapsed_seconds": round(time.time() - t0, 3)
        }
        yield f"data: {json.dumps(result_payload)}\n\n"

# Singleton instance
engine = EmbeddingGemmaEngine()
