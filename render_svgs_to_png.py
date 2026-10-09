import os
import shutil
from playwright.sync_api import sync_playwright

DIAGRAMS_DIR = r"C:\Users\kenhu\embeddinggemma2_demo\diagrams"
SCREENSHOTS_DIR = r"C:\Users\kenhu\embeddinggemma2_demo\screenshots"
ARTIFACT_DIR = r"C:\Users\kenhu\.gemini\antigravity-cli\brain\d034a62d-c064-4b67-94f5-74dcd94b0b6a\diagrams"
os.makedirs(DIAGRAMS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

svg_files = [
    ("diagram_01_modular_pyramid.svg", "diagram_01_modular_pyramid.png", 1200, 720),
    ("diagram_02_matryoshka_slicing.svg", "diagram_02_matryoshka_slicing.png", 1200, 720),
    ("diagram_03_asymmetric_prefix_flow.svg", "diagram_03_asymmetric_prefix_flow.png", 1200, 680),
]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    for svg_name, png_name, w, h in svg_files:
        svg_path = os.path.join(DIAGRAMS_DIR, svg_name)
        png_path = os.path.join(DIAGRAMS_DIR, png_name)
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #070a12;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
  }}
  svg {{
    display: block;
    width: {w}px;
    height: {h}px;
  }}
</style>
</head>
<body>
{svg_content}
</body>
</html>"""

        # 2x scale for Retina 2400px wide
        context = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=2.0)
        page = context.new_page()
        page.set_content(html_content)
        page.screenshot(path=png_path)
        context.close()

        # Copy to screenshots & artifacts
        shutil.copy2(png_path, os.path.join(SCREENSHOTS_DIR, png_name))
        shutil.copy2(png_path, os.path.join(ARTIFACT_DIR, png_name))
        shutil.copy2(svg_path, os.path.join(ARTIFACT_DIR, svg_name))
        print(f"Rendered {svg_name} -> {png_name} (2400x{h*2}px)")

    browser.close()
print("All professional SVG diagrams successfully rendered to high-res PNG!")
