import streamlit as st
import google.generativeai as genai
import os

# Configure API key (replace or use env var)
#GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or "your_api_key_here"
GEMINI_API_KEY = "testing-AIzaSyBPS9ZrdmIve3AzsH6iKKCplNMslzPxmAMtesting-"
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# App title
st.title("📞 Call Message Formatter (Gemini AI)")

# Input
user_input = st.text_input("Enter customer message (e.g., billing complaint):", 
                           "facing billing issues for my telephone connection")

# Process button
if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Generating response..."):
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
"{user_input}"
"""
            try:
                response = model.generate_content(prompt)
                st.subheader("📌 Output")
                st.code(response.text.strip(), language="text")
            except Exception as e:
                st.error(f"Error: {e}")
