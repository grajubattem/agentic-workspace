import vertexai
from google.generative_models import GenerativeModel

vertexai.init(project="your-project-id", location="us-central1")

model = GenerativeModel("gemini-1.5-flash")

call_transcript = """
Customer: Hi, I was charged twice on my last bill and I need a refund.
Representative: I'm sorry to hear that. Let me pull up your billing history to check.
"""

prompt = f"""
You are an AI that extracts intent from call center conversations.

Given this call transcript, return:
1. Customer Intent
2. Representative Intent

Transcript:
{call_transcript}

Format:
Customer Intent: <summary>
Representative Intent: <summary>
"""

response = model.generate_content(prompt)
print(response.text)
