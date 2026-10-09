import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    print("[run_server] Starting Uvicorn on 127.0.0.1:8088...")
    uvicorn.run("app.server:app", host="127.0.0.1", port=8088, log_level="info")
