import json
import os

import requests
import streamlit as st

st.set_page_config(page_title="Invoice Extractor")
st.title("Invoice Extractor")
st.caption("Upload an invoice or receipt")
uploaded = st.file_uploader("Document", type=["jpg", "jpeg", "png", "pdf", "txt"])

if uploaded and st.button("Extract JSON", type="primary"):
    endpoint = os.getenv("API_URL", "http://localhost:8000/extract")
    with st.spinner("Reading document with Ollama..."):
        try:
            result = requests.post(endpoint, files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}, timeout=180)
            result.raise_for_status()
            payload = result.json()
            field_rows = [
                {"field": field, "value": value}
                for field, value in payload.items()
                if field != "_confidence"
            ]
            st.dataframe(field_rows, hide_index=True, width="stretch")
            with st.expander("View JSON"):
                st.json(payload)
            st.download_button("Download JSON", json.dumps(payload, indent=2), "invoice.json", "application/json")
        except requests.HTTPError as error:
            try:
                detail = result.json().get("detail", result.text)
            except ValueError:
                detail = result.text
            st.error(f"Extraction failed: {detail}")
        except requests.ConnectionError:
            st.error("The extraction service is not running. Start the FastAPI server, then try again.")
        except requests.Timeout:
            st.error("Extraction timed out. Check that Ollama is running and try again.")
        except requests.RequestException as error:
            st.error(f"Extraction failed: {error}")