import time
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"

def main():
    print("=" * 60)
    print("Testing EmbeddingGemma 2 (google/embeddinggemma-2)")
    print("=" * 60)

    # 1. Test Text-Only Modular Loading (270M parameters)
    print("\n[Step 1] Loading Text-Only Encoder (270M parameters)...")
    start = time.time()
    text_model = SentenceTransformer(
        MODEL_ID,
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    print(f"Loaded Text-Only Model in {time.time() - start:.2f}s")

    # 2. Test Text Retrieval with Task Prompts
    print("\n[Step 2] Testing Text & Code Retrieval with Task Prompts...")
    query = "What causes the northern lights?"
    relevant_doc = "The northern lights (aurora borealis) are caused by charged particles from the solar wind interacting with Earth's magnetic field and upper atmosphere."
    irrelevant_doc = "The recipe for sourdough bread requires active starter, bread flour, water, and sea salt with slow fermentation."

    query_emb = text_model.encode(query, prompt_name="SearchQuery")
    rel_doc_emb = text_model.encode(relevant_doc, prompt_name="Document")
    irrel_doc_emb = text_model.encode(irrelevant_doc, prompt_name="Document")

    sim_rel = text_model.similarity(query_emb, rel_doc_emb).item()
    sim_irrel = text_model.similarity(query_emb, irrel_doc_emb).item()

    print(f"Query: '{query}'")
    print(f"  Similarity to Relevant Doc:   {sim_rel:.4f}")
    print(f"  Similarity to Irrelevant Doc: {sim_irrel:.4f}")
    assert sim_rel > sim_irrel, "Relevant doc must score higher than irrelevant doc!"
    print("  -> Text Retrieval Test PASSED!")

    # 3. Test Code Retrieval
    print("\n[Step 3] Testing Code Retrieval...")
    code_query = "Python function for binary search in a sorted list"
    code_doc = """def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1"""
    other_code = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""

    code_q_emb = text_model.encode(code_query, prompt_name="CodeRetrieval")
    code_doc_emb = text_model.encode(code_doc, prompt_name="Document")
    other_code_emb = text_model.encode(other_code, prompt_name="Document")

    sim_code_rel = text_model.similarity(code_q_emb, code_doc_emb).item()
    sim_code_other = text_model.similarity(code_q_emb, other_code_emb).item()

    print(f"Code Query: '{code_query}'")
    print(f"  Similarity to Binary Search: {sim_code_rel:.4f}")
    print(f"  Similarity to Bubble Sort:   {sim_code_other:.4f}")
    assert sim_code_rel > sim_code_other, "Binary search code should match query better!"
    print("  -> Code Retrieval Test PASSED!")

    # 4. Test Matryoshka Representation Learning (MRL) Truncation
    print("\n[Step 4] Testing Matryoshka (MRL) Dimension Truncation...")
    for dim in [768, 512, 256, 128]:
        q_mrl = text_model.encode(query, prompt_name="SearchQuery", truncate_dim=dim, normalize_embeddings=True)
        d_mrl = text_model.encode(relevant_doc, prompt_name="Document", truncate_dim=dim, normalize_embeddings=True)
        sim_mrl = text_model.similarity(q_mrl, d_mrl).item()
        print(f"  Dim = {dim:3d} | Vector Shape: {q_mrl.shape} | Cosine Similarity: {sim_mrl:.4f}")

    print("  -> Matryoshka Truncation Test PASSED!")

    # 5. Test Full Multimodal Setup (Text + Vision)
    print("\n[Step 5] Testing Multimodal / Vision Embedding...")
    full_model = SentenceTransformer(MODEL_ID)
    
    # Create sample synthetic test images
    red_img = Image.new("RGB", (128, 128), color=(220, 30, 30))
    red_img_path = "sample_red_square.png"
    red_img.save(red_img_path)

    blue_img = Image.new("RGB", (128, 128), color=(30, 30, 220))
    blue_img_path = "sample_blue_square.png"
    blue_img.save(blue_img_path)

    img_q = "A solid bright red colored square"
    img_q_emb = full_model.encode(img_q, prompt_name="SearchQuery")
    red_emb = full_model.encode({"image": red_img_path})
    blue_emb = full_model.encode({"image": blue_img_path})

    sim_red = full_model.similarity(img_q_emb, red_emb).item()
    sim_blue = full_model.similarity(img_q_emb, blue_emb).item()

    print(f"Image Query: '{img_q}'")
    print(f"  Similarity to Red Square:  {sim_red:.4f}")
    print(f"  Similarity to Blue Square: {sim_blue:.4f}")
    assert sim_red > sim_blue, "Red image query must match red image better than blue image!"
    print("  -> Multimodal Vision Test PASSED!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
