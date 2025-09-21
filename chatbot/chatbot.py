# chatbot/chatbot_app.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="VotEdge Chatbot", layout="wide")
st.title("VotEdge Chatbot 🤖")
st.markdown("Ask questions about political data (news, tweets, sentiment analysis).")

# Load CSV data
data_folder = Path("data/raw")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    st.warning("No CSV data found in the data folder!")
    st.stop()

# Combine CSVs into a list of rows for simple keyword search
all_rows = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        all_rows.append({"file": csv_file.name, "text": " ".join(row.astype(str).tolist())})

# Helper: search rows for keywords
def search_rows(question, rows):
    keywords = question.lower().split()
    results = []
    for r in rows:
        if any(k in r["text"].lower() for k in keywords):
            results.append(f"{r['file']}: {r['text']}")
    return results[:5]  # return top 5 matches

# Streamlit input
user_question = st.text_input("Ask your question about political data:")

if st.button("Get Answer") and user_question:
    with st.spinner("Searching for answer..."):
        matches = search_rows(user_question, all_rows)
        if matches:
            st.markdown("**Answer (from your data):**")
            for m in matches:
                st.markdown(f"- {m}")
        else:
            st.markdown("No matching data found.")
