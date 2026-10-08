import os
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = r"C:\Users\kenhu\embeddinggemma2_demo\screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def run_tests():
    print("[1/6] Launching Playwright browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        print("[2/6] Navigating to http://127.0.0.1:8088...")
        page.goto("http://127.0.0.1:8088", wait_until="networkidle")
        page.wait_for_selector("#tab-btn-compare")

        # ----------------------------------------------------
        # Test 1: Pairwise Matcher & MRL Inspector (Text vs Document)
        # ----------------------------------------------------
        print("[3/6] Testing Pairwise Matcher (Text Query vs Document)...")
        page.click("#btn-run-compare")
        page.wait_for_function("() => !document.getElementById('compare-result-box').classList.contains('hidden')", timeout=30000)
        time.sleep(1.0)
        path1 = os.path.join(SCREENSHOTS_DIR, "01_pairwise_matcher_mrl.png")
        page.screenshot(path=path1)
        print(f"  Saved: {path1}")

        # ----------------------------------------------------
        # Test 2: Cross-Modal Pairwise Match (Text Query vs Red Image)
        # ----------------------------------------------------
        print("[4/6] Testing Cross-Modal Text Query vs Red Image...")
        page.fill("#compare-a-text", "Bright crimson red square graphic polygon")
        page.click("button:has-text('Red Image')")
        time.sleep(0.5)
        page.click("#btn-run-compare")
        
        # Wait for completion (allow 40s for image encoding on CPU)
        page.wait_for_function("() => document.getElementById('compare-stream-pct').innerText === '100%'", timeout=60000)
        time.sleep(1.5)
        path2 = os.path.join(SCREENSHOTS_DIR, "02_crossmodal_vision_match.png")
        page.screenshot(path=path2)
        print(f"  Saved: {path2}")

        # ----------------------------------------------------
        # Test 3: Multimodal Search & RAG
        # ----------------------------------------------------
        print("[5/6] Testing Multimodal Search & RAG Tab...")
        page.click("#tab-btn-search")
        page.wait_for_selector("#btn-run-search")
        time.sleep(0.5)
        
        # Run search for Ocean waves
        page.click("#btn-run-search")
        
        # Wait for search results grid to have cards populated
        page.wait_for_function("() => document.getElementById('search-results-grid').children.length > 0", timeout=30000)
        time.sleep(1.5)
        path3 = os.path.join(SCREENSHOTS_DIR, "03_multimodal_search_rag.png")
        page.screenshot(path=path3)
        print(f"  Saved: {path3}")

        # ----------------------------------------------------
        # Test 4: Vector & Interleaved Studio
        # ----------------------------------------------------
        print("[6/6] Testing Vector & Interleaved Studio Tab...")
        page.click("#tab-btn-embed")
        page.wait_for_selector("#btn-run-embed")
        time.sleep(0.5)
        
        page.click("#btn-run-embed")
        
        # Wait for vector output dimension badge to show 768
        page.wait_for_function("() => document.getElementById('embed-dim').innerText.includes('768')", timeout=30000)
        time.sleep(1.0)
        path4 = os.path.join(SCREENSHOTS_DIR, "04_vector_studio.png")
        page.screenshot(path=path4)
        print(f"  Saved: {path4}")

        # ----------------------------------------------------
        # Test 5: Model Architecture Tab
        # ----------------------------------------------------
        print("Testing Model Architecture Tab...")
        page.click("#tab-btn-architecture")
        time.sleep(0.8)
        path5 = os.path.join(SCREENSHOTS_DIR, "05_architecture_matrix.png")
        page.screenshot(path=path5)
        print(f"  Saved: {path5}")

        # ----------------------------------------------------
        # Test 6: Modular Switcher Modal
        # ----------------------------------------------------
        print("Testing Modular Switcher Modal...")
        page.evaluate("openConfigModal()")
        time.sleep(0.8)
        path6 = os.path.join(SCREENSHOTS_DIR, "06_modular_switcher_modal.png")
        page.screenshot(path=path6)
        print(f"  Saved: {path6}")

        browser.close()
        print("\nAll 6 UI screenshots and feature tests completed successfully!")

if __name__ == "__main__":
    run_tests()
