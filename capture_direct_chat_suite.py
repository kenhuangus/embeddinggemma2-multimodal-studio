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

def wait_for_stream_done(page, expected_completed_count, timeout=60000):
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
    print("[1/4] Launching Chromium browser with 1.5x scaling...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1450, "height": 950}, device_scale_factor=1.5)
        page = context.new_page()

        page.goto("http://127.0.0.1:8088", wait_until="networkidle")
        page.wait_for_selector("#tab-btn-chat")
        page.click("button:has-text('Clear Chat')")
        time.sleep(0.5)

        # ----------------------------------------------------
        # TURN 1: Prompt & Model Direct Response
        # ----------------------------------------------------
        print("[2/4] Turn 1: Explaining Matryoshka Representation Learning...")
        prompt_1 = "Can you explain Matryoshka Representation Learning (MRL) and why it allows 6x compression down to 128d?"
        page.fill("#chat-input", prompt_1)
        page.click("#btn-send-chat")

        wait_for_stream_done(page, 1, timeout=30000)
        capture_panel(page, "direct_chat_turn_01_mrl.png")

        # ----------------------------------------------------
        # TURN 2: Multi-Turn Follow-Up Prompt & Response
        # ----------------------------------------------------
        print("[3/4] Turn 2: Follow-up code request...")
        prompt_2 = "How do I truncate embeddings to 128 dimensions in Python using SentenceTransformers?"
        page.fill("#chat-input", prompt_2)
        page.click("#btn-send-chat")

        wait_for_stream_done(page, 2, timeout=35000)
        capture_panel(page, "direct_chat_turn_02_python.png")

        # ----------------------------------------------------
        # TURN 3: Multi-Turn Follow-Up Tradeoffs & Scaling
        # ----------------------------------------------------
        print("[4/4] Turn 3: Follow-up tradeoff analysis...")
        prompt_3 = "What are the main tradeoffs or failure modes when deploying 128d truncated vectors at scale?"
        page.fill("#chat-input", prompt_3)
        page.click("#btn-send-chat")

        wait_for_stream_done(page, 3, timeout=40000)
        capture_panel(page, "direct_chat_turn_03_tradeoffs.png")

        browser.close()
        print("\nAll direct multi-turn conversational chat screenshots captured successfully!")

if __name__ == "__main__":
    run()
