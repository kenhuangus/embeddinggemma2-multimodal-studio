import os
import json
import numpy as np
from app.model_engine import engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_ASSETS_DIR = os.path.join(BASE_DIR, "sample_assets")
CACHE_FILE = os.path.join(BASE_DIR, "app", "cached_corpus_vectors.json")

corpus_specs = [
    {
        "id": "item-1",
        "modality": "text",
        "content": "The northern lights (aurora borealis) are caused by charged particles from the solar wind colliding with atmospheric gases in Earth's magnetosphere, producing glowing emerald and violet curtains.",
        "prompt": "Document"
    },
    {
        "id": "item-2",
        "modality": "code",
        "content": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
        "prompt": "Document"
    },
    {
        "id": "item-3",
        "modality": "code",
        "content": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
        "prompt": "Document"
    },
    {
        "id": "item-4",
        "modality": "image",
        "content": os.path.join(SAMPLE_ASSETS_DIR, "red_square.png"),
        "prompt": None
    },
    {
        "id": "item-5",
        "modality": "image",
        "content": os.path.join(SAMPLE_ASSETS_DIR, "ocean_blue.png"),
        "prompt": None
    },
    {
        "id": "item-6",
        "modality": "audio",
        "content": os.path.join(SAMPLE_ASSETS_DIR, "deep_bell.wav"),
        "prompt": None
    },
    {
        "id": "item-7",
        "modality": "audio",
        "content": os.path.join(SAMPLE_ASSETS_DIR, "ocean_waves.wav"),
        "prompt": None
    },
    {
        "id": "item-8",
        "modality": "video",
        "content": os.path.join(SAMPLE_ASSETS_DIR, "motion_demo.mp4"),
        "prompt": None
    }
]

def precompute():
    print("Loading model engine...")
    engine.load_model("full")
    cached = {}
    for spec in corpus_specs:
        print(f"Pre-encoding {spec['id']} ({spec['modality']})...")
        vec = engine.encode_single(spec["modality"], spec["content"], spec["prompt"], truncate_dim=768)
        cached[spec["id"]] = [float(x) for x in vec]
        print(f"  Done: {spec['id']} (dim={len(vec)})")
    
    with open(CACHE_FILE, "w") as f:
        json.dump(cached, f)
    print(f"Successfully saved {len(cached)} cached vectors to {CACHE_FILE}")

if __name__ == "__main__":
    precompute()
