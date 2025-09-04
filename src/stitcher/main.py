import uvicorn
from src.stitcher import app

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import STITCHER_URL, STITCHER_PORT

if __name__ == "__main__":
    uvicorn.run(app, host=STITCHER_URL, port=STITCHER_PORT)
