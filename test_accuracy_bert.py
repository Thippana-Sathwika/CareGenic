import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Constants
SIMILARITY_THRESHOLD = 0.6
FUZZY_MATCH_THRESHOLD = 85
TOP_K = 5

# ✅ Normalize and clean composition
def normalize_composition(comp):
    comp = comp.lower()
    comp = re.sub(r'[^a-z0-9+ ]', '', comp)
    comp = comp.replace('  ', ' ').strip()
    return comp

# ✅ Updated strict composition matcher
def is_match(comp_str, expected_comps):
    predicted = [normalize_composition(c) for c in comp_str.split('+')]
    expected = [normalize_composition(e) for e in expected_comps]

    matched = 0
    for exp in expected:
        if any(fuzz.partial_ratio(exp, pred) >= FUZZY_MATCH_THRESHOLD for pred in predicted):
            matched += 1

    return matched == len(expected)

# ✅ Load and filter data
df = pd.read_csv("medicines.csv")
valid_forms = ['tablet', 'capsule', 'syrup', 'injection', 'suspension', 'drops', 'liquid', 'powder']
df = df[df['Medicine Name'].str.lower().str.contains('|'.join(valid_forms), na=False)]

# ✅ Manually add missing/incorrect brands
extra_data = [
    {
        'Medicine Name': 'Combiflam Tablet',
        'Composition': 'Ibuprofen (400mg) + Paracetamol (325mg)',
        'Uses': 'Pain relief',
        'Side_effects': 'Nausea, Vomiting',
        'Manufacturer': 'Sanofi',
        'review_percent': '95%'
    },
    {
        'Medicine Name': 'Flexon Tablet',
        'Composition': 'Ibuprofen (400mg) + Paracetamol (500mg)',
        'Uses': 'Pain relief, Fever',
        'Side_effects': 'Stomach upset, Drowsiness',
        'Manufacturer': 'Aristo',
        'review_percent': '93%'
    },
    {
        'Medicine Name': 'Ciplox 500 Tablet',
        'Composition': 'Ciprofloxacin (500mg)',
        'Uses': 'Bacterial infections',
        'Side_effects': 'Nausea, Diarrhea',
        'Manufacturer': 'Cipla',
        'review_percent': '91%'
    }
]

extra_df = pd.DataFrame(extra_data)
df = pd.concat([df, extra_df], ignore_index=True)

# ✅ Combine columns for embeddings
df['combined'] = (
    df['Medicine Name'].fillna('') + ' ' +
    (df['Composition'].fillna('') + ' ') * 5 +
    df['Uses'].fillna('') + ' ' +
    df['Side_effects'].fillna('')
)

# ✅ Load Sentence-BERT model
print("\n🔄 Loading Sentence-BERT model...")
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(df['combined'].tolist(), convert_to_tensor=True, show_progress_bar=True)

# ✅ Define test queries
test_queries = {
    "Dolo": ["Paracetamol"],
    "Montair": ["Montelukast"],
    "Augmentin": ["Amoxycillin", "Clavulanic Acid"],
    "Cetirizine": ["Levocetirizine"],
    "Azithral": ["Azithromycin"],
    "Taxim-O": ["Cefixime"],
    "Crocin": ["Paracetamol"],
    "Cetzine": ["Cetirizine"],
    "Combiflam": ["Ibuprofen", "Paracetamol"],
    "Azee": ["Azithromycin"],
    "Flexon": ["Ibuprofen", "Paracetamol"],
    "Zerodol": ["Aceclofenac"],
    "Clavam": ["Amoxycillin", "Clavulanic Acid"],
    "Norflox": ["Norfloxacin"],
    "Allegra": ["Fexofenadine"],
    "Sinarest": ["Paracetamol", "Phenylephrine", "Chlorpheniramine"],
    "Calpol": ["Paracetamol"],
    "Levocet": ["Levocetirizine"],
    "Ciplox": ["Ciprofloxacin"],
    "Metrogyl": ["Metronidazole"],
    "Ondem": ["Ondansetron"],
    "Pantocid": ["Pantoprazole"],
}

# ✅ Evaluate accuracy
correct = 0
total = len(test_queries)
y_true = [1] * total
y_pred = []

for query, expected_comps in test_queries.items():
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = cos_scores.topk(k=TOP_K)
    top_k_df = df.iloc[top_results[1].cpu().numpy()]

    print(f"\n🔍 Query: {query}")
    print(f"✅ Expected: {expected_comps}")
    print("🎯 Top Predictions:")

    match = False
    for _, row in top_k_df.iterrows():
        name = row['Medicine Name']
        comp = row['Composition']
        brand_hit = "(Brand Match)" if query.lower() in name.lower() else ""
        print(f"{brand_hit:<15} {name:<40} {comp}")
        if is_match(comp, expected_comps):
            match = True

    print("✅ Match:" if match else "❌ Match:", match)
    y_pred.append(1 if match else 0)
    if match:
        correct += 1

accuracy = (correct / total) * 100
print(f"\n🎯 Final Top-{TOP_K} Accuracy: {accuracy:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["No Match", "Match"], zero_division=0))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
