"""
predict.py
Inference module for Indian Citizen & Consumer Complaint Intelligence & Routing System.

How it works (3 simple steps):
1. Load the pre-trained Sentence Transformer encoder + trained classifier
2. Convert the complaint text into a 384-dim dense vector (embedding)
3. Classify the department, analyze urgency, and compute routing queue
"""

import os
import joblib
import nltk
import numpy as np
import re
from sentence_transformers import SentenceTransformer      # Dense multilingual encoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────
# Setup VADER sentiment analyzer (for urgency detection)
# ──────────────────────────────────────────────────────────────
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

# ──────────────────────────────────────────────────────────────
# File paths for saved models
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLF_PATH = os.path.join(MODELS_DIR, "complaint_classifier.joblib")
ENCODER_NAME_PATH = os.path.join(MODELS_DIR, "encoder_model_name.txt")

# ──────────────────────────────────────────────────────────────
# Global variables to hold loaded models (loaded once, reused)
# ──────────────────────────────────────────────────────────────
_encoder = None
_classifier = None


def get_models():
    """
    Load the Sentence Transformer encoder and the trained classifier.
    Models are loaded only once and then reused for every prediction.
    """
    global _encoder, _classifier

    if _encoder is None or _classifier is None:
        # Check that model files exist
        if not os.path.exists(CLF_PATH) or not os.path.exists(ENCODER_NAME_PATH):
            raise FileNotFoundError(
                f"Model files missing in {MODELS_DIR}. Please run `python train.py` first."
            )

        # Read which encoder model was used during training
        with open(ENCODER_NAME_PATH, "r") as f:
            model_name = f.read().strip()

        # Load the Sentence Transformer encoder (downloads on first use, then cached)
        print(f"Loading Sentence Transformer: {model_name}")
        _encoder = SentenceTransformer(model_name)

        # Load the trained MLP neural network classifier
        _classifier = joblib.load(CLF_PATH)
        print("Models loaded successfully!")

    return _encoder, _classifier


# ──────────────────────────────────────────────────────────────
# Indian civic and consumer contextual urgency keywords
# ──────────────────────────────────────────────────────────────
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
    """
    Determine how urgent a complaint is based on:
    - Sentiment score (negative = more urgent)
    - Presence of urgency keywords
    """
    text_lower = text.lower()

    # Check which keywords are found in the complaint
    detected_critical = [k for k in CRITICAL_KEYWORDS if k in text_lower]
    detected_high = [k for k in HIGH_KEYWORDS if k in text_lower]
    detected_medium = [k for k in MEDIUM_KEYWORDS if k in text_lower]

    # Decide urgency level
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
    """Map department + urgency to a specific triage queue."""
    if urgency == "Critical":
        return f"{department} - Escalation Queue"
    elif urgency == "High":
        return f"{department} - Priority Queue"
    elif urgency == "Medium":
        return f"{department} - Standard Queue"
    else:
        return f"{department} - General Queue"

def clean_hinglish_text(text: str) -> str:
    """
    Standardize common informal Hinglish texting abbreviations and phonetic spellings.
    Examples:
      'aa rha'  -> 'aa raha'
      'nhi'     -> 'nahi'
      'bohot'   -> 'bahut'
      'pani'    -> 'paani'
    """
    subs = {
        r"\bnhi\b": "nahi",
        r"\bni\b": "nahi",
        r"\brha\b": "raha",
        r"\brhi\b": "rahi",
        r"\brhe\b": "rahe",
        r"\baarha\b": "aa raha",
        r"\baarhi\b": "aa rahi",
        r"\baarahe\b": "aa rahe",
        r"\baarah\b": "aa raha",
        r"\bbohot\b": "bahut",
        r"\bbht\b": "bahut",
        r"\bpani\b": "paani",
        r"\bpe\b": "par",
        r"\bplz\b": "please",
        r"\bpls\b": "please",
        r"\bkr\b": "kar",
        r"\bkro\b": "karo",
        r"\bkre\b": "kare"
    }
    normalized = text
    for pattern, replacement in subs.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


# ──────────────────────────────────────────────────────────────
# Department Anchor Keywords for High-Precision Routing
# ──────────────────────────────────────────────────────────────
DEPARTMENT_KEYWORDS = {
    "Water Supply & Drainage": [
        "paani", "pani", "water", "pipeline", "pipe", "drain", "drainage", "gutter",
        "sewer", "sewerline", "nallah", "nala", "nali", "nal", "nalke", "borewell",
        "tanker"
    ],
    "Electricity & Power": [
        "bijli", "bijlee", "light", "electricity", "power", "powercut", "power cut",
        "voltage", "transformer", "pole", "wire", "current", "load shedding", "khamba"
    ],
    "Roads & Civic Infrastructure": [
        "sadak", "sarak", "road", "pothole", "potholes", "gaddha", "gaddhe",
        "footpath", "flyover", "signal", "traffic signal", "speed breaker", "divider", "sinkhole"
    ],
    "Garbage & Sanitation": [
        "kachra", "kooda", "kuda", "garbage", "waste", "trash", "dustbin", "safai",
        "ghanta gadi", "badbu", "dead animal", "sanitation"
    ],
    "Public Transport": [
        "bus", "metro", "auto", "rickshaw", "conductor", "driver", "bus stop", "bus stand"
    ],
    "Citizen & Consumer Services": [
        "parcel", "order", "delivery", "refund", "replace", "application", "portal",
        "pension", "ration card", "aadhaar", "certificate", "license", "grievance"
    ]
}


def predict(text: str) -> dict:
    """
    Main prediction function. Takes a complaint string and returns:
    - department: which department should handle this
    - confidence: how confident the model is (0 to 1)
    - urgency: Low / Medium / High / Critical
    - routing_queue: which triage queue to send it to
    - sentiment_score: VADER compound sentiment (-1 to +1)
    - detected_cues: which urgency keywords were found
    - breakdown: confidence scores for ALL departments
    """
    # Handle empty input
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

    # Standardize informal Hinglish texting shorthand
    cleaned_text = clean_hinglish_text(text)

    # ── Step 1: Load models ──
    encoder, classifier = get_models()

    # ── Step 2: Convert complaint text to a dense 384-dim vector ──
    # The encoder understands the MEANING of the text
    text_vector = encoder.encode([cleaned_text], normalize_embeddings=True)     # shape: (1, 384)

    # ── Step 3: Combine Neural Probabilities with Domain Anchor Keywords ──
    raw_probs = list(classifier.predict_proba(text_vector)[0])
    classes = list(classifier.classes_)

    # Detect matched keywords in the cleaned text
    text_lower = cleaned_text.lower()
    keyword_matches = {}
    for dept, kws in DEPARTMENT_KEYWORDS.items():
        count = sum(1 for kw in kws if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
        if count > 0:
            keyword_matches[dept] = count

    # If unambiguous domain keywords are present, apply strong prior boost
    if len(keyword_matches) == 1:
        matched_dept = list(keyword_matches.keys())[0]
        boosted_probs = []
        for cls_name, prob in zip(classes, raw_probs):
            if cls_name == matched_dept:
                boosted_probs.append(prob + 5.0)
            else:
                boosted_probs.append(prob * 0.05)
        total_p = sum(boosted_probs)
        probabilities = [p / total_p for p in boosted_probs]
    elif len(keyword_matches) > 1:
        boosted_probs = []
        for cls_name, prob in zip(classes, raw_probs):
            weight = keyword_matches.get(cls_name, 0)
            boosted_probs.append(prob + weight * 2.0)
        total_p = sum(boosted_probs)
        probabilities = [p / total_p for p in boosted_probs]
    else:
        probabilities = raw_probs

    best_idx = int(np.argmax(probabilities))
    predicted_dept = str(classes[best_idx])
    confidence = float(probabilities[best_idx])

    # ── Step 4: Analyze sentiment for urgency ──
    sentiment_scores = sia.polarity_scores(text)
    compound = sentiment_scores["compound"]

    # ── Step 5: Determine urgency and routing queue ──
    urgency, detected_cues = determine_urgency(text, compound)
    routing_queue = compute_routing_queue(predicted_dept, urgency, detected_cues)

    # ── Step 6: Build the breakdown of all department scores ──
    breakdown = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classes, probabilities)
    }

    return {
        "department": predicted_dept,
        "predicted_department": predicted_dept,
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "urgency": urgency,
        "urgency_level": urgency,
        "routing_queue": routing_queue,
        "routing_action": routing_queue,
        "sentiment_score": round(compound, 3),
        "sentiment": "Negative" if compound <= -0.1 else ("Positive" if compound >= 0.1 else "Neutral"),
        "detected_cues": detected_cues,
        "breakdown": breakdown
    }

# Alias for notebook and external scripts
predict_complaint = predict




# ──────────────────────────────────────────────────────────────
# Quick test: Run this file directly to test 10 sample complaints
# ──────────────────────────────────────────────────────────────
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

    print("Testing 10 Indian user complaints:\n")
    for text in sample_complaints:
        r = predict(text)
        print(f"Text: {text}")
        print(f" -> Dept: {r['department']} ({r['confidence_pct']}) | Urgency: {r['urgency']}")
        print(f" -> Queue: {r['routing_queue']}\n")
