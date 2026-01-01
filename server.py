from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import run_chat

app = FastAPI()

# Enable CORS (Allows your React app on port 3000 to talk to this on port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust if your react app is on a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data format coming from React
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Pass the message to the LangGraph agent
        response_text = run_chat(request.message)
        return {"response": response_text}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)