import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("medicines.csv")

# Fill missing values
df['Uses'] = df['Uses'].fillna('')
df['Side_effects'] = df['Side_effects'].fillna('')
df['Composition'] = df['Composition'].fillna('')

# Combine relevant columns for semantic understanding
df['combined'] = (
    df['Medicine Name'].astype(str) + ' ' +
    df['Uses'].astype(str) + ' ' +
    df['Side_effects'].astype(str) + ' ' +
    df['Composition'].astype(str)
)

# Clean text
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['combined'] = df['combined'].apply(clean_text)

# Load BERT model
print("\n🔄 Loading BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("\n🔄 Generating embeddings for all medicines...")
embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)

# Store the embeddings to avoid recomputation (optional)
np.save("medicine_embeddings.npy", embeddings)

# Recommendation function
def recommend_medicines_bert(query, top_n=5):
    query = clean_text(query)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Medicine Name', 'Uses', 'Side_effects', 'Composition']]

# --- Test it ---
input_medicine = "Azithral 500 Tablet"
print(f"\n🔍 Top recommendations for: {input_medicine}\n")
recommendations = recommend_medicines_bert(input_medicine)
print(recommendations.to_string(index=False))
