# Cross-Category-Sentiment-Analysis-on-Amazon-product-reviews
# 🧠 Cross-Category Sentiment Analysis on Amazon Product Reviews

### 🔍 Overview
This project explores **cross-domain sentiment analysis** on **Amazon product reviews**, where models trained on one product category (e.g., Electronics) are tested on another (e.g., Books).  
It combines **classical machine learning** and **deep contextual embeddings (BERT)** to create a **hybrid feature representation** that improves model adaptability across product domains.

---

### 🎯 Objective
- Analyze and classify customer sentiments from Amazon reviews.  
- Evaluate **four ML models** — Naive Bayes, Logistic Regression, SVM, and Random Forest.  
- Compare performance of **TF-IDF** vs **Hybrid (TF-IDF + BERT)** embeddings.  
- Demonstrate **real-time sentiment prediction** using a **Streamlit web app**.  
- Test **cross-category performance** (Electronics → Books).

---

### 🧩 Methodology

1. **Data Collection**
   - `amazon_Electronic_review.csv` → Training data  
   - `book_review.csv` → Testing data  

2. **Data Preprocessing**
   - Remove nulls, neutral reviews (rating = 3)
   - Encode sentiments (Positive = 1, Negative = 0)

3. **Feature Extraction**
   - TF-IDF Vectorization  
   - SentenceTransformer BERT embeddings (`all-MiniLM-L6-v2`)  
   - Hybrid representation using `scipy.hstack()`

4. **Model Training**
   - Models used: Naive Bayes, Logistic Regression, SVM, Random Forest  
   - Evaluation metrics: Accuracy, Precision, Recall, F1-score

5. **Cross-Domain Testing**
   - Train on Electronics dataset, test on Books dataset.

6. **Visualization**
   - Accuracy and F1-score comparison bar charts  
   - Real-time prediction and model comparison using Streamlit.

---

### ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/Kamalkishor2003/Cross-Category-Sentiment-Analysis-on-Amazon-product-reviews.git
cd Cross-Category-Sentiment-Analysis-on-Amazon-product-reviews

# Install dependencies
pip install -r requirements.txt

👨‍💻 Author

KamalKishor Sunwasiya
M.Sc. Data Science — 2025
Project: Cross-Category Sentiment Analysis Using Machine Learning
📧 Email: [kamalkishor.sunwasiya2003@gmail.com]
🌐 GitHub: [github.com/Kamalkishor2003]
