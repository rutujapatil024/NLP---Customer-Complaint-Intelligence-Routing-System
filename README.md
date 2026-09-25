# Citizen Grievance AI — Complaint Intelligence & Routing System

An intelligent NLP intake and triage system that classifies raw citizen and consumer complaints, determines operational urgency using sentiment analysis and keyword cues, and automatically routes issues to the responsible public department and triage queue.

## 🚀 Key Features

1. **TF-IDF Feature Extraction**: Sublinear term-frequency inverse document frequency vectorizer (`ngram_range=(1,2)`, `max_features=5000`).
2. **Multi-Class Department Classifier**: Calibrated Logistic Regression model trained across 6 departments:
   - `Water Supply & Drainage`
   - `Electricity & Power`
   - `Roads & Civic Infrastructure`
   - `Garbage & Sanitation`
   - `Public Transport`
   - `Citizen & Consumer Services`
3. **Hybrid Urgency Engine**:
   - Pretrained **VADER Sentiment Analyzer** (`compound` polarity score).
   - Domain-specific **Urgency Keyword Cue Scanners** (e.g., `immediate`, `severe`, `dangerous`, `accident`, `asap`, `health risk`, `overflowing`, `sparks`).
   - Categorized into 4 urgency tiers: **Low**, **Medium**, **High**, **Critical**.
4. **Automated Queue Dispatch Rules**: Combines predicted department and urgency into actionable queues (e.g., `Roads & Civic Infrastructure - Escalation Queue`, `Water Supply & Drainage - Priority Queue`).
5. **Modern Citizen Portal UI**: Built with responsive layouts, department icons, real-time confidence gauge, quick-test scenario presets, and an AI analysis diagnostic panel.

---

## 🛠️ Project Structure

```text
Customer Complaint Intelligence/
├── data/
│   └── complaints_dataset.csv       # Labeled dataset (900 rows across 6 categories)
├── models/
│   ├── tfidf_vectorizer.joblib      # Serialized scikit-learn TF-IDF vectorizer
│   └── complaint_classifier.joblib  # Serialized Logistic Regression model
├── static/
│   ├── index.html                   # Support desk intake UI
│   ├── style.css                    # SaaS design system & status styling
│   └── app.js                       # Frontend triage controller & API integration
├── train.py                         # Dataset generator & model training pipeline
├── predict.py                       # Single predict(text) inference module
├── app.py                           # Thin FastAPI endpoint & static file host
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1. (Optional) Re-train the Model
The model is already pre-trained and saved in `models/`. If you want to re-generate synthetic complaints and re-train:
```bash
python train.py
```

### 2. Run the Intake Web Server
Launch the FastAPI backend server:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Open the Demo in your Browser
Navigate to:
```
http://127.0.0.1:8000
```
- Click any of the **Quick Test Scenarios** pills or pick a complaint from the **Recent Complaints Stream** sidebar.
- Click **Analyze Complaint** (or press <kbd>Ctrl</kbd> + <kbd>Enter</kbd>).
- Watch the **Department Badge**, **Urgency Level**, **Model Confidence**, and **Target Routing Queue** update in real-time.

---

## 🧪 Direct Python API Usage

You can also use the core inference module directly in Python scripts:

```python
from predict import predict

result = predict("I was charged twice for $89 on invoice #9401! Please cancel immediately.")

print(result["department"])     # "Billing"
print(result["confidence_pct"]) # "87.0%"
print(result["urgency"])        # "High"
print(result["routing_queue"])  # "Billing - Priority Billing Escalation"
```
