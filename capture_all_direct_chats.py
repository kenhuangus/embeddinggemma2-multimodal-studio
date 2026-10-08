import os
import time
import shutil
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = r"C:\Users\kenhu\embeddinggemma2_demo\screenshots"
STATIC_SCREENSHOTS = r"C:\Users\kenhu\embeddinggemma2_demo\app\static\screenshots"
ARTIFACT_SCREENSHOTS = r"C:\Users\kenhu\.gemini\antigravity-cli\brain\d034a62d-c064-4b67-94f5-74dcd94b0b6a\screenshots"

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(STATIC_SCREENSHOTS, exist_ok=True)
os.makedirs(ARTIFACT_SCREENSHOTS, exist_ok=True)

def wait_for_stream_done(page, expected_completed_count, timeout=45000):
    page.wait_for_function(
        f"() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length >= {expected_completed_count}",
        timeout=timeout
    )
    time.sleep(1.0)

def capture_panel(page, filename):
    # Temporarily expand chat container so all turns and code blocks are fully rendered without scroll clipping
    page.evaluate("""() => {
        const c = document.getElementById('chat-messages');
        if (c) {
            c.style.minHeight = 'auto';
            c.style.maxHeight = 'none';
            c.style.height = 'auto';
            c.style.overflow = 'visible';
        }
        window.scrollTo(0, 0);
    }""")
    time.sleep(0.5)

    dest_local = os.path.join(SCREENSHOTS_DIR, filename)
    page.locator("#tab-chat").screenshot(path=dest_local)

    # Revert styling
    page.evaluate("""() => {
        const c = document.getElementById('chat-messages');
        if (c) {
            c.style.minHeight = '460px';
            c.style.maxHeight = '580px';
            c.style.height = '';
            c.style.overflowY = 'auto';
            c.scrollTop = c.scrollHeight;
        }
    }""")

    # Sync to static and artifacts
    for d in [STATIC_SCREENSHOTS, ARTIFACT_SCREENSHOTS]:
        shutil.copy2(dest_local, os.path.join(d, filename))

    print(f"Captured: {filename}")

def run():
    print("[1/5] Launching Chromium browser with 1.5x scaling...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1450, "height": 950}, device_scale_factor=1.5)
        page = context.new_page()

        page.goto("http://127.0.0.1:8088", wait_until="networkidle")
        page.wait_for_selector("#tab-btn-chat")
        time.sleep(1.0)

        # ----------------------------------------------------
        # TEST 1: Pure Multi-Turn Text Chat (Turns 1, 2, 3)
        # ----------------------------------------------------
        print("[2/5] Capturing Multi-Turn Text Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        # Turn 1
        page.evaluate("loadSampleChatPrompt('explain_mrl')")
        time.sleep(0.3)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=30000)
        capture_panel(page, "direct_chat_turn_01_mrl.png")
        shutil.copy2(os.path.join(SCREENSHOTS_DIR, "direct_chat_turn_01_mrl.png"), os.path.join(SCREENSHOTS_DIR, "chat_01_text_panel.png"))

        # Turn 2
        page.evaluate("loadSampleChatPrompt('code_example')")
        time.sleep(0.3)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 2, timeout=35000)
        capture_panel(page, "direct_chat_turn_02_python.png")

        # Turn 3
        page.fill("#chat-input", "What are the key production tradeoffs or failure modes when deploying 128d truncated vectors at scale?")
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 3, timeout=40000)
        capture_panel(page, "direct_chat_turn_03_tradeoffs.png")

        # ----------------------------------------------------
        # TEST 2: Audio Voice Chat
        # ----------------------------------------------------
        print("[3/5] Capturing Audio Voice Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        page.evaluate("loadSampleChatPrompt('voice_match')")
        time.sleep(0.3)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=35000)
        capture_panel(page, "chat_02_audio_panel.png")

        # ----------------------------------------------------
        # TEST 3: Video Upload & Chat
        # ----------------------------------------------------
        print("[4/5] Capturing Video Upload & Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        page.evaluate("loadSampleChatPrompt('video_match')")
        time.sleep(0.3)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=40000)
        capture_panel(page, "chat_03_video_panel.png")

        # ----------------------------------------------------
        # TEST 4: PDF Document Ingestion & Chat
        # ----------------------------------------------------
        print("[5/5] Capturing PDF Document Chat...")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        page.evaluate("loadSampleChatPrompt('pdf_spec')")
        time.sleep(0.3)
        page.click("#btn-send-chat")
        wait_for_stream_done(page, 1, timeout=40000)
        capture_panel(page, "chat_04_pdf_panel.png")

        # Sync all panel screenshots
        for f in ["chat_01_text_panel.png", "chat_02_audio_panel.png", "chat_03_video_panel.png", "chat_04_pdf_panel.png"]:
            src = os.path.join(SCREENSHOTS_DIR, f)
            for d in [STATIC_SCREENSHOTS, ARTIFACT_SCREENSHOTS]:
                shutil.copy2(src, os.path.join(d, f))

        browser.close()
        print("\nAll direct multi-turn and multimodal chat screenshots captured successfully!")

if __name__ == "__main__":
    run()
