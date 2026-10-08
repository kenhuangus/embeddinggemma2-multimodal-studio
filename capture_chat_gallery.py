import os
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = r"C:\Users\kenhu\embeddinggemma2_demo\screenshots"
STATIC_SCREENSHOTS = r"C:\Users\kenhu\embeddinggemma2_demo\app\static\screenshots"
ARTIFACT_SCREENSHOTS = r"C:\Users\kenhu\.gemini\antigravity-cli\brain\d034a62d-c064-4b67-94f5-74dcd94b0b6a\screenshots"

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(STATIC_SCREENSHOTS, exist_ok=True)
os.makedirs(ARTIFACT_SCREENSHOTS, exist_ok=True)

def wait_for_stream_done(page, expected_actions_count, timeout=80000):
    page.wait_for_function(
        f"() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length >= {expected_actions_count}",
        timeout=timeout
    )
    time.sleep(1.5)

def expand_and_capture(page, base_name):
    # Expand chat-messages so all messages, attachments, video/audio players, and citations are fully visible
    page.evaluate("""() => {
        const c = document.getElementById('chat-messages');
        if (c) {
            c.style.height = 'auto';
            c.style.maxHeight = 'none';
            c.style.overflow = 'visible';
        }
        window.scrollTo(0, 0);
    }""")
    time.sleep(0.5)

    panel_path = os.path.join(SCREENSHOTS_DIR, f"{base_name}_panel.png")
    full_path = os.path.join(SCREENSHOTS_DIR, f"{base_name}_full.png")

    page.locator("#tab-chat").screenshot(path=panel_path)
    page.screenshot(path=full_path, full_page=True)

    # Revert styling for continuous interaction
    page.evaluate("""() => {
        const c = document.getElementById('chat-messages');
        if (c) {
            c.style.height = '480px';
            c.style.maxHeight = '';
            c.style.overflowY = 'auto';
        }
    }""")

    # Copy to static and artifacts
    for target_dir in [STATIC_SCREENSHOTS, ARTIFACT_SCREENSHOTS]:
        import shutil
        shutil.copy2(panel_path, os.path.join(target_dir, f"{base_name}_panel.png"))
        shutil.copy2(full_path, os.path.join(target_dir, f"{base_name}_full.png"))

    print(f"  Captured & replicated {base_name}: panel & full")

def capture_chat_suite():
    print("[1/5] Launching browser with high-DPI viewport...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 950}, device_scale_factor=1.5)
        page = context.new_page()

        page.goto("http://127.0.0.1:8088", wait_until="networkidle")
        page.wait_for_selector("#tab-btn-chat")
        time.sleep(1.0)

        # -----------------------------------------------------------------
        # SCENARIO 1: Pure Text Chat & Semantic Retrieval
        # -----------------------------------------------------------------
        print("[2/5] Running Scenario 1: Text Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        page.fill("#chat-input", "Explain Matryoshka Representation Learning (MRL) and how EmbeddingGemma 2 maintains 98.4% retrieval accuracy at 128 dimensions.")
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=40000)
        expand_and_capture(page, "chat_01_text")

        # -----------------------------------------------------------------
        # SCENARIO 2: Audio Voice Chat & Acoustic Ingestion
        # -----------------------------------------------------------------
        print("[3/5] Running Scenario 2: Audio Voice Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        page.click("button:has-text('Test Audio Chat')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=45000)
        expand_and_capture(page, "chat_02_audio")

        # -----------------------------------------------------------------
        # SCENARIO 3: Video Upload & Motion Dynamics Analysis
        # -----------------------------------------------------------------
        print("[4/5] Running Scenario 3: Video Upload & Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        page.click("button:has-text('Analyze Video Clip')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=75000)
        expand_and_capture(page, "chat_03_video")

        # -----------------------------------------------------------------
        # SCENARIO 4: PDF Document Ingestion & Page Citations
        # -----------------------------------------------------------------
        print("[5/5] Running Scenario 4: PDF Document Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        page.click("button:has-text('Load Spec PDF')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=60000)
        expand_and_capture(page, "chat_04_pdf")

        # Also update the original numbered files 07, 08, 09, 10 for consistency
        import shutil
        shutil.copy2(os.path.join(SCREENSHOTS_DIR, "chat_01_text_panel.png"), os.path.join(SCREENSHOTS_DIR, "07_text_chat_streaming.png"))
        shutil.copy2(os.path.join(SCREENSHOTS_DIR, "chat_02_audio_panel.png"), os.path.join(SCREENSHOTS_DIR, "08_audio_chat_streaming.png"))
        shutil.copy2(os.path.join(SCREENSHOTS_DIR, "chat_03_video_panel.png"), os.path.join(SCREENSHOTS_DIR, "09_video_chat_streaming.png"))
        shutil.copy2(os.path.join(SCREENSHOTS_DIR, "chat_04_pdf_panel.png"), os.path.join(SCREENSHOTS_DIR, "10_pdf_chat_streaming.png"))

        for target_dir in [STATIC_SCREENSHOTS, ARTIFACT_SCREENSHOTS]:
            shutil.copy2(os.path.join(SCREENSHOTS_DIR, "07_text_chat_streaming.png"), os.path.join(target_dir, "07_text_chat_streaming.png"))
            shutil.copy2(os.path.join(SCREENSHOTS_DIR, "08_audio_chat_streaming.png"), os.path.join(target_dir, "08_audio_chat_streaming.png"))
            shutil.copy2(os.path.join(SCREENSHOTS_DIR, "09_video_chat_streaming.png"), os.path.join(target_dir, "09_video_chat_streaming.png"))
            shutil.copy2(os.path.join(SCREENSHOTS_DIR, "10_pdf_chat_streaming.png"), os.path.join(target_dir, "10_pdf_chat_streaming.png"))

        browser.close()
        print("\nAll chat screenshots re-captured and synchronized with full visibility!")

if __name__ == "__main__":
    capture_chat_suite()
