import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import webbrowser

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Define a mapping of known sites (can be expanded)
known_sites = {
    "google": "https://www.google.com/",
    "youtube": "https://www.youtube.com/",
    "facebook": "https://www.facebook.com/",
    "instagram": "https://www.instagram.com/",
    "github": "https://www.github.com/",
    "openai": "https://www.openai.com/",
    "gmail": "https://mail.google.com/"
}

st.set_page_config(page_title="Smart Web Search Agent", layout="centered")

st.title("🔍 Smart Web Search Agent")

search_input = st.text_input("Enter a website name", placeholder="e.g., google")

if search_input:
    query = search_input.lower().strip()

    # Check in predefined list
    if query in known_sites:
        url = known_sites[query]
        st.success(f"✅ Found: [{url}]({url})")
        if st.button("Open Website"):
            webbrowser.open_new_tab(url)
    else:
        # Ask Gemini for the link
        prompt = f"Give me the official website link for '{query}' if it exists. Just give the URL."
        try:
            response = model.generate_content(prompt)
            website_url = response.text.strip()

            # Basic validation
            if website_url.startswith("http"):
                st.success(f"🌐 Found: [{website_url}]({website_url})")
                if st.button("Open Website"):
                    webbrowser.open_new_tab(website_url)
            else:
                st.error("❌ Page Not Found")
        except Exception as e:
            st.error(f"Error: {e}")
