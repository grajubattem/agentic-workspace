import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import webbrowser

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

st.set_page_config(page_title="Support Assistant", layout="centered")
st.title("🤖 AI Support Assistant")

user_issue = st.text_area("Describe your issue (e.g., 'billing issue with Amazon'):", height=150)

if st.button("🔍 Get Help Links"):
    with st.spinner("Getting helpful resources..."):
        prompt = f"""
        The user is describing an issue: "{user_issue}"

        Your task is to provide 3 to 5 official or helpful URLs (only links!) that can help solve the issue.
        These must be clickable links related to help pages, support forums, contact pages, etc.
        Provide only valid and complete links, no descriptions.
        """

        try:
            response = model.generate_content(prompt)
            raw_links = response.text

            # Extract links from Gemini response
            links = re.findall(r'(https?://[^\s]+)', raw_links)

            if links:
                st.subheader("🔗 Helpful Links")
                for link in links:
                    st.markdown(f"- [{link}]({link})")
            else:
                st.warning("No links found. Try refining the issue.")

        except Exception as e:
            st.error(f"Gemini Error: {e}")
