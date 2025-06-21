import google.generativeai as genai

# Configure the Gemini API
genai.configure(api_key="AIzaSyCLqjkhHSm4nk7-fjEC13gGpo_otRO50kQ")

# Initialize the Gemini Pro model
model = genai.GenerativeModel("gemini-1.5-flash")

# Sample call center conversation
call_transcript = """
Customer: Hi, I was charged twice on my last bill and I need a refund.
Representative: I'm sorry to hear that. Let me pull up your billing history to check.
"""

# Prompt template
prompt = f"""
You are an AI assistant that extracts intent from call center conversations.

Given the following transcript, extract:
1. Customer Intent
2. Representative Intent

Return the output in JSON format with keys: customer_intent, representative_intent.

Transcript:
{call_transcript}
"""

# Generate response
response = model.generate_content(prompt)

# Print result
print(response.text)
