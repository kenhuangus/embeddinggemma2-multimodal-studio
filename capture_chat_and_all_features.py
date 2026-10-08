import os
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = r"C:\Users\kenhu\embeddinggemma2_demo\screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def run_chat_and_ui_tests():
    print("[1/5] Launching browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        print("[2/5] Navigating to http://127.0.0.1:8088...")
        page.goto("http://127.0.0.1:8088", wait_until="networkidle")
        page.wait_for_selector("#tab-btn-chat")

        # ----------------------------------------------------
        # Test 1: Text Chat with Streaming RAG
        # ----------------------------------------------------
        print("[3/5] Testing Text Chat...")
        page.fill("#chat-input", "How does Matryoshka Representation Learning (MRL) achieve 6x storage compression at 128d?")
        page.click("#btn-send-chat")
        
        # Wait for assistant response to reach 100% or show Synthesized badge
        page.wait_for_function("() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length > 0", timeout=30000)
        time.sleep(1.0)
        
        path_text = os.path.join(SCREENSHOTS_DIR, "07_text_chat_streaming.png")
        page.screenshot(path=path_text)
        print(f"  Saved: {path_text}")

        # ----------------------------------------------------
        # Test 2: Audio Voice Chat
        # ----------------------------------------------------
        print("[4/5] Testing Audio Voice Chat...")
        page.click("button:has-text('Test Audio Chat')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        
        # Wait for completion
        page.wait_for_function("() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length >= 2", timeout=35000)
        time.sleep(1.5)
        
        path_audio = os.path.join(SCREENSHOTS_DIR, "08_audio_chat_streaming.png")
        page.screenshot(path=path_audio)
        print(f"  Saved: {path_audio}")

        # ----------------------------------------------------
        # Test 3: Video Upload & Chat
        # ----------------------------------------------------
        print("[5/5] Testing Video Upload & Chat...")
        page.click("button:has-text('Analyze Video Clip')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        
        # Wait for completion
        page.wait_for_function("() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length >= 3", timeout=45000)
        time.sleep(1.5)
        
        path_video = os.path.join(SCREENSHOTS_DIR, "09_video_chat_streaming.png")
        page.screenshot(path=path_video)
        print(f"  Saved: {path_video}")

        # ----------------------------------------------------
        # Test 4: PDF Ingestion & Chat
        # ----------------------------------------------------
        print("Testing PDF Ingestion & Chat...")
        page.click("button:has-text('Load Spec PDF')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        
        # Wait for completion
        page.wait_for_function("() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length >= 4", timeout=35000)
        time.sleep(1.5)
        
        path_pdf = os.path.join(SCREENSHOTS_DIR, "10_pdf_chat_streaming.png")
        page.screenshot(path=path_pdf)
        print(f"  Saved: {path_pdf}")

        browser.close()
        print("\nAll Chat features (Text, Audio, Video, PDF) tested and captured successfully!")

if __name__ == "__main__":
    run_chat_and_ui_tests()
