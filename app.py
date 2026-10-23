"""
app.py
Thin FastAPI application serving the Customer Complaint Intelligence & Routing System demo.
Endpoints:
- POST /api/predict: Invokes predict(text)
- GET /api/info: Returns system metadata, categories, and queues
- Static files served from ./static/
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict

app = FastAPI(
    title="Customer Complaint Intelligence & Routing API",
    description="Multilingual Sentence Transformer NLP system for complaint classification and urgency routing",
    version="2.0.0"
)

# Enable CORS for local testing flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComplaintRequest(BaseModel):
    text: str

@app.post("/api/predict")
def predict_endpoint(payload: ComplaintRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")
    
    try:
        result = predict(payload.text)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/info")
def get_info():
    return {
        "status": "active",
        "system": "Indian Citizen & Consumer Complaint Intelligence & Routing Engine",
        "model_type": "Multilingual Sentence Transformer (MiniLM) + MLP Neural Classifier",
        "urgency_analyzer": "NLTK VADER + Contextual Safety Cues",
        "departments": [
            "Water Supply & Drainage",
            "Electricity & Power",
            "Roads & Civic Infrastructure",
            "Garbage & Sanitation",
            "Public Transport",
            "Citizen & Consumer Services"
        ],
        "urgency_tiers": ["Low", "Medium", "High", "Critical"]
    }

# Mount static files directory to serve the single-page customer support intake UI
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

