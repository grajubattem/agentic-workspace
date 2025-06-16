import pandas as pd
import google.generativeai as genai
import streamlit as st

st.title("📊 Gemini Data Explorer")

genai.configure(api_key="tingAIzaSyBaUnw0ZDWet0jlABvdMCQ7MhTOEfmSnm8")
#model = genai.GenerativeModel("gemini-pro")
model = genai.GenerativeModel("gemini-2.0-flash")

file = st.file_uploader("Upload your CSV", type="csv")

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your dataset:")

    if question:
        prompt = f"""
        You are a data analysis assistant. Given the DataFrame below, answer the following question:
        {question}

        DataFrame (first 10 rows):
        {df.head(10).to_csv(index=False)}
        """

        response = model.generate_content(prompt)
        st.write(response.text)
