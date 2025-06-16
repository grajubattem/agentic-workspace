import streamlit as st
from vz_agent import classify_input

st.set_page_config(page_title="Smart Classifier Agent", layout="wide")
st.title("🔍 Agentic Classifier App")

# User Input
user_text = st.text_area("Enter any message or query")

if st.button("Classify & Show Layout"):
    category = classify_input(user_text)
    st.success(f"🧠 Detected Category: `{category}`")

    # Dynamic Layout Switching
    if category == "finance":
        st.header("💰 Finance Dashboard")
        st.line_chart([100, 200, 150, 300])
        st.metric("Market Trend", "+3.2%", "+1.4%")

    elif category == "health":
        st.header("🩺 Health Insights")
        st.progress(0.7)
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/88/Heart_rate_graph.png", width=400)

    elif category == "education":
        st.header("📚 Education Center")
        st.code("print('Hello, student!')", language="python")
        st.info("Tip: Practice 30 mins daily.")

    elif category == "tech":
        st.header("💻 Tech Trends")
        st.json({"AI": "Exploding", "Quantum": "Emerging", "Cloud": "Mainstream"})

    else:
        st.warning("🤔 Unknown category or couldn't classify.")
