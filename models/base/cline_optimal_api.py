from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ollama
import uvicorn
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Cline-Optimal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str
    temperature: Optional[float] = 0.2

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "cline-optimal:latest",
        "endpoints": {
            "/chat": "Send messages to cline-optimal",
            "/docs": "API documentation"
        }
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    response = ollama.chat(
        model="cline-optimal:latest",
        messages=[{"role": "user", "content": request.text}],
        options={"temperature": request.temperature}
    )
    
    return {
        "response": response["message"]["content"],
        "model": "cline-optimal:latest"
    }

if __name__ == "__main__":
    print("Starting Cline-Optimal API on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
