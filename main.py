from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY:", GEMINI_API_KEY)
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=" + GEMINI_API_KEY

# In-memory conversation history (for simplicity; use a database for persistence across sessions)
conversation_history = []

class PromptRequest(BaseModel):
    prompt: str
    image_base64: str = None

@app.post("/split")
async def split_bill(data: PromptRequest):
    global conversation_history
    try:
        # System instruction (only sent once at the start)
        if not conversation_history:
            conversation_history.append({
                "role": "user",
                "parts": [{
                    "text": (
                        "You are a helpful assistant that ONLY responds to bill-splitting or expense-sharing related questions. You can convert currency but only if the user asks to convert it."
                        "Always give concise answers."
                        "If the user asks anything unrelated to bills, expenses, or money-splitting, politely respond with:\n\n"
                        "'I'm only able to assist with bill-splitting and expense-sharing questions. Please ask something related to that.'"
                    )
                }]
            })

        # Add the current user message to history
        user_message = {"role": "user", "parts": [{"text": data.prompt}]}
        if data.image_base64:
            user_message["parts"].append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": data.image_base64
                }
            })
        conversation_history.append(user_message)

        # Send the full history to Gemini
        res = requests.post(
            GEMINI_ENDPOINT,
            json={
                "contents": conversation_history,
                "generationConfig": {
                    "temperature": 0.5,
                    "topK": 1,
                    "topP": 1
                }
            }
        )

        result = res.json()

        if "candidates" in result:
            bot_response = result["candidates"][0]["content"]["parts"][0]["text"]
            # Add bot response to history
            conversation_history.append({"role": "model", "parts": [{"text": bot_response}]})
            return {"response": bot_response}
        else:
            return {"error": result}

    except Exception as e:
        return {"error": str(e)}

# Optional: Reset history endpoint (for testing or new sessions)
@app.post("/reset")
async def reset_history():
    global conversation_history
    conversation_history = []