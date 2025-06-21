from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os

# Gemini API key setup
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or "your_api_key_here"
GEMINI_API_KEY = "testing-AIzaSyBPS9ZrdmIve3AzsH6iKKCplNMslzPxmAMtesting-"
genai.configure(api_key=GEMINI_API_KEY)

# Load model
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app
app = FastAPI(title="Call Center Intent Extractor API")

# Request model
class MessageRequest(BaseModel):
    message: str

# Response model
class IntentResponse(BaseModel):
    customer: str
    representative: str

@app.post("/case-followup", response_model=IntentResponse)
def generate_intent(request: MessageRequest):
    prompt = f"""
You are an AI model that reformats customer call inputs into two simple outputs:
1. A short customer message starting with "customer: "
2. A short representative message starting with "representative: "

Format:
customer: "..."
representative: "..."

Example Input: 
"facing billing issues for my telephone connection"

Expected Output:
customer: "custmer asked about billing issues"
representative: "reprentative shared about billing related answer"

Now process this:
"{request.message}"
"""
    try:
        response = model.generate_content(prompt)
        raw_output = response.text.strip()

        # Parse expected format
        lines = raw_output.split("\n")
        customer = ""
        representative = ""
        for line in lines:
            if line.lower().startswith("customer:"):
                customer = line.split(":", 1)[1].strip().strip('"')
            elif line.lower().startswith("representative:"):
                representative = line.split(":", 1)[1].strip().strip('"')

        if not customer or not representative:
            raise ValueError("Invalid output format from Gemini")

        return {"customer": customer, "representative": representative}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
