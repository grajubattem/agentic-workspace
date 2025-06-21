import streamlit as st
import google.generativeai as genai
import os

# Set your Gemini API key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDNmGzvvPpENIFA0Cf10IWCOf3e_kSe9OY"
genai.configure(api_key='AIzaSyDNmGzvvPpENIFA0Cf10IWCOf3e_kSe9OY')

# Initialize Gemini Pro
model = genai.GenerativeModel("gemini-1.5-flash")

# Title
st.title("📞 Call Center Intent Extractor using Gemini AI")

# Text input
transcript = st.text_area("Enter call center transcript:", height=200, placeholder="""
Customer: Hi, I was charged twice on my last bill and I need a refund.
Representative: I'm sorry to hear that. Let me pull up your billing history to check.
""")

# Button to process
if st.button("Extract Intents"):
    if not transcript.strip():
        st.warning("Please enter a call transcript.")
    else:
        with st.spinner("Analyzing intents..."):
            prompt = f"""
            You are an AI assistant that extracts intent from call center conversations.

            Given the following transcript, extract:
            1. Customer Intent
            2. Representative Intent

            Return the output in JSON format with keys: customer_intent, representative_intent.

            Transcript:
            {transcript}
            """
            try:
                response = model.generate_content(prompt)
                st.subheader("📌 Extracted Intents")
                st.json(response.text)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
