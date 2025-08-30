import uvicorn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from config import CONTROLLER_URL, CONTROLLER_PORT

if __name__ == "__main__":
    uvicorn.run(
        "src.controller.controller_app:app",

        reload=True
    )
