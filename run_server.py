"""Start the local Invest AI FastAPI server."""

import uvicorn

from invest_ai.api import api


if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)
