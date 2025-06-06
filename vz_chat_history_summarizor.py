import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Setup
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # Flash 2.0 also works here

st.set_page_config(page_title="Chat Summarizer", layout="wide")
st.title("🧠 Gemini Flash Chat History Summarizer")

# Sample chat or user upload
sample_chat = """
User: Hey, I want to book a flight from Delhi to Bangalore.
Bot: Sure, when do you plan to travel?
User: This Friday.
Bot: Morning or evening?
User: Morning. And window seat please.
Bot: Great, I've booked it for 8:00 AM with a window seat.
User: Also, add a reminder to pack.
"""

chat_input = st.text_area("Paste your chat history here:", value=sample_chat, height=250)

summary_style = st.radio("Summarize as:", ["📝 Bullet Summary", "✅ Decisions/Actions", "💬 FAQ-style"])

if st.button("Summarize with Gemini Flash"):
    prompt = f"""
    Summarize this multi-turn chat between a user and an assistant.

    Style: {summary_style}

    Chat:
    {chat_input}
    """

    with st.spinner("Analyzing with Gemini Flash 2.0..."):
        response = model.generate_content(prompt)
        st.subheader("📌 Summary")
        st.markdown(response.text)
