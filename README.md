# CareGenic# 🧠 Intelligent Medicine Recommendation System

An AI-powered medicine recommendation system that helps users find suitable medications based on symptoms or partial medicine names. This system combines **Sentence-BERT (SBERT)** embeddings for semantic similarity with **fuzzy string matching** on compositions to provide accurate results even with ambiguous inputs.

## 📌 Project Highlights

- ✅ Dataset of 11,000+ medicines scraped from **1mg**
- ✅ SBERT-based semantic search for symptom and medicine queries
- ✅ Fuzzy matching on medicine compositions
- ✅ Combined scoring mechanism for top-N recommendations
- ✅ Fast and scalable predictions
- ✅ Web interface or command-line support

---

## 🗂️ Dataset Description

The dataset includes the following fields:

| Field             | Description                             |
|------------------|-----------------------------------------|
| `Medicine Name`  | Name of the medicine                    |
| `Composition`    | Active ingredients                      |
| `Uses`           | Primary medical usage                   |
| `Side_effects`   | Known side effects                      |
| `Image URL`      | Thumbnail of medicine                   |
| `Manufacturer`   | Company manufacturing the medicine      |
| `Review %`       | Positive review percentage              |

---

## 🧠 Tech Stack

- **Python** (Core logic)
- **Sentence-BERT (SBERT)** from `sentence-transformers`
- **FuzzyWuzzy** for string similarity (composition matching)
- **Pandas**, **Numpy** for data handling
- **Streamlit** / Flask (optional) for web UI

---

## 🚀 How It Works

1. **Data Preprocessing**
   - Clean text, remove duplicates
   - Normalize composition strings

2. **Model Preparation**
   - Encode all medicine names and uses with SBERT
   - Store embeddings for fast search

3. **Prediction Pipeline**
   - User enters a query (symptom or partial medicine name)
   - Compute SBERT similarity with all records
   - Perform fuzzy matching on compositions
   - Combine both scores to rank results

4. **Output**
   - Display top-N relevant medicines with:
     - Name
     - Use
     - Composition
     - Image (optional)
     - Similarity score

---

## 🧪 Example Usage

```bash
# Inference script
python predict.py --query "fever and body pain"
