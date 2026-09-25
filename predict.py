"""
predict.py
Inference module for Indian Citizen & Consumer Complaint Intelligence & Routing System.
Predicts category, calculates urgency using VADER sentiment and Indian contextual cues,
and outputs deterministic routing queues.
"""

import os
import re
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH = os.path.join(MODELS_DIR, "complaint_classifier.joblib")

_vectorizer = None
_classifier = None

def get_models():
    global _vectorizer, _classifier
    if _vectorizer is None or _classifier is None:
        if not os.path.exists(VEC_PATH) or not os.path.exists(CLF_PATH):
            raise FileNotFoundError(
                f"Model files missing in {MODELS_DIR}. Please run `python train.py` first."
            )
        _vectorizer = joblib.load(VEC_PATH)
        _classifier = joblib.load(CLF_PATH)
    return _vectorizer, _classifier

# Indian civic and consumer contextual urgency keywords
CRITICAL_KEYWORDS = [
    "accident", "electrocution", "fire", "sparking", "hospital", "emergency",
    "police", "safety hazard", "danger", "dangerous", "dharna", "consumer court",
    "burst", "collapsed", "sinkhole", "contaminated", "dengue", "suffocating",
    "dead animal", "life threatening", "severe accident", "mishap", "casualty"
]

HIGH_KEYWORDS = [
    "immediate action", "immediately", "urgent", "urgently", "as soon as possible",
    "overflowing", "unhygienic", "power cut", "darkness", "potholes", "no water",
    "foul smell", "bad smell", "blocked", "choked", "stagnant", "suffering",
    "struggling", "damage", "triple", "severe hardship"
]

MEDIUM_KEYWORDS = [
    "several days", "two days", "past week", "delay", "not resolved",
    "kindly arrange", "please check", "pending", "not working", "damaged",
    "broken", "taking too long", "proper update", "not arriving", "frequent",
    "leaking", "cleaning", "repair"
]

def determine_urgency(text: str, compound_sentiment: float):
    text_lower = text.lower()
    
    detected_critical = [k for k in CRITICAL_KEYWORDS if k in text_lower]
    detected_high = [k for k in HIGH_KEYWORDS if k in text_lower]
    detected_medium = [k for k in MEDIUM_KEYWORDS if k in text_lower]
    
    if len(detected_critical) > 0 or (compound_sentiment <= -0.65 and len(detected_high) > 0):
        urgency = "Critical"
    elif compound_sentiment <= -0.40 or len(detected_high) > 0:
        urgency = "High"
    elif compound_sentiment <= -0.10 or len(detected_medium) > 0:
        urgency = "Medium"
    else:
        urgency = "Low"
        
    all_detected = detected_critical + detected_high + detected_medium
    return urgency, all_detected

def compute_routing_queue(department: str, urgency: str, detected_cues: list) -> str:
    if urgency == "Critical":
        return f"{department} - Escalation Queue"
    elif urgency == "High":
        return f"{department} - Priority Queue"
    elif urgency == "Medium":
        return f"{department} - Standard Queue"
    else:
        return f"{department} - General Queue"

def predict(text: str) -> dict:
    if not text or not text.strip():
        return {
            "department": "Citizen & Consumer Services",
            "confidence": 0.0,
            "confidence_pct": "0.0%",
            "urgency": "Low",
            "routing_queue": "Citizen Portal - General Triage Desk",
            "sentiment_score": 0.0,
            "detected_cues": [],
            "breakdown": {}
        }
        
    vectorizer, classifier = get_models()
    
    features = vectorizer.transform([text])
    probabilities = classifier.predict_proba(features)[0]
    classes = classifier.classes_
    
    best_idx = probabilities.argmax()
    predicted_dept = str(classes[best_idx])
    confidence = float(probabilities[best_idx])
    
    sentiment_scores = sia.polarity_scores(text)
    compound = sentiment_scores["compound"]
    
    urgency, detected_cues = determine_urgency(text, compound)
    routing_queue = compute_routing_queue(predicted_dept, urgency, detected_cues)
    
    breakdown = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classes, probabilities)
    }
    
    return {
        "department": predicted_dept,
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "urgency": urgency,
        "routing_queue": routing_queue,
        "sentiment_score": round(compound, 3),
        "detected_cues": detected_cues,
        "breakdown": breakdown
    }

if __name__ == "__main__":
    sample_complaints = [
        "The water supply has not been coming properly for the last two days. Please look into this issue as soon as possible.",
        "There is a lot of garbage near our building and it has not been collected for several days. Kindly arrange for cleaning.",
        "The street light in our area has not been working for the past week. It becomes very difficult to travel at night. Please fix it.",
        "The road near our society is damaged and has many potholes. Please repair it to avoid accidents.",
        "I have submitted my application but have not received any update yet. Kindly check the status and provide an update.",
        "There is frequent power supply interruption in our area. Please check the issue and resolve it.",
        "The drainage is overflowing near our building, causing a bad smell and unhygienic conditions. Please take immediate action.",
        "I received the wrong item in my order. Please arrange for a replacement or refund.",
        "The bus did not arrive on time today, and there was no information about the delay. Please look into this.",
        "The service is taking too long and my complaint has still not been resolved. Kindly provide a proper update."
    ]
    
    print("Testing 10 Indian user complaints:")
    for text in sample_complaints:
        r = predict(text)
        print(f"\nText: {text}")
        print(f" -> Dept: {r['department']} ({r['confidence_pct']}) | Urgency: {r['urgency']}")
        print(f" -> Queue: {r['routing_queue']}")
