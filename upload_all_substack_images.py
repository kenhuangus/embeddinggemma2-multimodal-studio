import os
import json
from substack import Api

sid = os.getenv("SUBSTACK_SID")
url = os.getenv("SUBSTACK_URL", "https://kenhuangus.substack.com")

api = Api(
    publication_url=url,
    cookies_string=f"substack.sid={sid}"
)

# Load existing mapping if present
mapping_path = r"C:\Users\kenhu\embeddinggemma2_demo\uploaded_chat_images.json"
mapping = {}
if os.path.exists(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

images_to_upload = [
    # Professional SVG Figures
    ("diagrams/diagram_01_modular_pyramid.png", r"C:\Users\kenhu\embeddinggemma2_demo\diagrams\diagram_01_modular_pyramid.png"),
    ("diagrams/diagram_02_matryoshka_slicing.png", r"C:\Users\kenhu\embeddinggemma2_demo\diagrams\diagram_02_matryoshka_slicing.png"),
    ("diagrams/diagram_03_asymmetric_prefix_flow.png", r"C:\Users\kenhu\embeddinggemma2_demo\diagrams\diagram_03_asymmetric_prefix_flow.png"),
    # Pristine Multi-Turn and Multimodal Chat Screenshots
    ("screenshots/direct_chat_turn_01_mrl.png", r"C:\Users\kenhu\embeddinggemma2_demo\screenshots\direct_chat_turn_01_mrl.png"),
    ("screenshots/chat_01_text_panel.png", r"C:\Users\kenhu\embeddinggemma2_demo\screenshots\chat_01_text_panel.png"),
    ("screenshots/direct_chat_turn_02_python.png", r"C:\Users\kenhu\embeddinggemma2_demo\screenshots\direct_chat_turn_02_python.png"),
    ("screenshots/direct_chat_turn_03_tradeoffs.png", r"C:\Users\kenhu\embeddinggemma2_demo\screenshots\direct_chat_turn_03_tradeoffs.png"),
    ("screenshots/chat_02_audio_panel.png", r"C:\Users\kenhu\embeddinggemma2_demo\screenshots\chat_02_audio_panel.png"),
    ("screenshots/chat_03_video_panel.png", r"C:\Users\kenhu\embeddinggemma2_demo\screenshots\chat_03_video_panel.png"),
    ("screenshots/chat_04_pdf_panel.png", r"C:\Users\kenhu\embeddinggemma2_demo\screenshots\chat_04_pdf_panel.png"),
]

for key, file_path in images_to_upload:
    print(f"Uploading {key} ({os.path.getsize(file_path)} bytes)...")
    try:
        res = api.get_image(file_path)
        img_url = res.get("url")
        if img_url:
            mapping[key] = img_url
            print(f"  -> SUCCESS: {img_url}")
        else:
            print(f"  -> FAILED: no url in response {res}")
    except Exception as e:
        print(f"  -> ERROR uploading {key}: {e}")

with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)

print("\nWrote updated mapping to", mapping_path)
