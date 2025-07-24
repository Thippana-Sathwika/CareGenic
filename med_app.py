import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Medicine Recommendation System", layout="centered")

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("medicines.csv")  # 🔁 Replace with your actual filename
    df.fillna('', inplace=True)  # Fill missing text fields
    df['combined'] = df['Uses'] + ' ' + df['Side_effects'] + ' ' + df['Composition']
    return df

df = load_data()

# Vectorize text
@st.cache_resource
def get_tfidf_matrix(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combined'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = get_tfidf_matrix(df)

# Streamlit UI
st.markdown("## 💊 Medicine Recommendation System")
st.markdown("Enter a **symptom**, **use-case**, or **medicine name** to find similar medicines based on **composition**, **usage**, and **side effects**.")

query = st.text_input("🔎 Search by medicine, symptom or condition", placeholder="e.g., dry cough, fever, paracetamol")

if query:
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[::-1][:5]

    if similarity[top_indices[0]] < 0.1:
        st.error("❌ No close matches found. Try a different symptom or spelling.")
    else:
        st.success("✅ Found similar medicines:")
        for i in top_indices:
            med = df.iloc[i]
            st.markdown(f"### 🩺 {med['Medicine Name']}")
            if med['Image URL']:
                st.image(med['Image URL'], width=200)
            st.write(f"**Composition:** {med['Composition']}")
            st.write(f"**Uses:** {med['Uses']}")
            st.write(f"**Side Effects:** {med['Side_effects']}")
            st.write(f"**Manufacturer:** {med['Manufacturer']}")
            st.progress(int(med['Excellent Review %']))
            st.markdown("---")
