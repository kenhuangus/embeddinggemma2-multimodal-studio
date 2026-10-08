import os
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = r"C:\Users\kenhu\embeddinggemma2_demo\screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def run():
    print("Launching browser for Video & PDF Chat captures...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        page.goto("http://127.0.0.1:8088", wait_until="networkidle")
        page.wait_for_selector("#tab-btn-chat")

        # ----------------------------------------------------
        # Test 1: Video Chat
        # ----------------------------------------------------
        print("Testing Video Chat...")
        page.click("button:has-text('Analyze Video Clip')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        
        # Wait for video message action footer to show
        page.wait_for_function(
            "() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length >= 1",
            timeout=75000
        )
        time.sleep(1.5)
        path_video = os.path.join(SCREENSHOTS_DIR, "09_video_chat_streaming.png")
        page.screenshot(path=path_video)
        print(f"Saved: {path_video}")

        # ----------------------------------------------------
        # Test 2: PDF Chat
        # ----------------------------------------------------
        print("Testing PDF Chat...")
        page.click("button:has-text('Load Spec PDF')")
        time.sleep(0.5)
        page.click("#btn-send-chat")
        
        # Wait for PDF message action footer to show
        page.wait_for_function(
            "() => document.querySelectorAll('#chat-messages div[id$=\"-actions\"]:not(.hidden)').length >= 2",
            timeout=75000
        )
        time.sleep(1.5)
        path_pdf = os.path.join(SCREENSHOTS_DIR, "10_pdf_chat_streaming.png")
        page.screenshot(path=path_pdf)
        print(f"Saved: {path_pdf}")

        browser.close()
        print("\nVideo and PDF Chat captures completed successfully!")

if __name__ == "__main__":
    run()
