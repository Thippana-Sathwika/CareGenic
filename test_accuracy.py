import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# 🔹 Step 1: Load and clean data
df = pd.read_csv("medicines.csv")

# 🔹 Step 2: Fill missing values
df['Composition'] = df['Composition'].fillna('')
df['Medicine Name'] = df['Medicine Name'].fillna('')

# 🔹 Step 3: Create TF-IDF matrix on 'Composition'
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Composition'])

# 🔹 Step 4: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, df['Medicine Name'], test_size=0.2, random_state=42
)

# 🔹 Step 5: Define accuracy function (Top-K accuracy)
def compute_topk_accuracy(X_test, y_test, tfidf_matrix, medicine_names, top_k=5):
    correct = 0
    total = len(y_test)

    for i in range(total):
        test_vec = X_test[i]

        # Compare test vector with full matrix
        similarities = cosine_similarity(test_vec, tfidf_matrix).flatten()

        # Get top-k most similar medicine indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_meds = medicine_names.iloc[top_indices].tolist()

        # Check if actual medicine is in top-k
        actual = y_test.iloc[i]
        if actual in top_meds:
            correct += 1

    return (correct / total) * 100

# 🔹 Step 6: Run accuracy test
topk_accuracy = compute_topk_accuracy(X_test, y_test, tfidf_matrix, df['Medicine Name'], top_k=5)
print(f"\n🎯 Top-5 Recommendation Accuracy: {topk_accuracy:.2f}%")
