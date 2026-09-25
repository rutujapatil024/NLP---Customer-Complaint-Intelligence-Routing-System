"""
train.py
Synthetic labeled dataset generator and model trainer for Indian Citizen & Consumer Complaints.
Categories:
- Water Supply & Drainage
- Electricity & Power
- Roads & Civic Infrastructure
- Garbage & Sanitation
- Public Transport
- Citizen & Consumer Services
"""

import os
import random
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

random.seed(42)

# Templates modeling natural Indian English complaints
COMPLAINT_TEMPLATES = {
    "Water Supply & Drainage": [
        "The water supply has not been coming properly for the last two days. Please look into this issue as soon as possible.",
        "The drainage is overflowing near our building, causing a bad smell and unhygienic conditions. Please take immediate action.",
        "Drinking water pipeline is leaking in our lane near house number {num}. Huge amount of water is being wasted.",
        "No water supply in our society since yesterday morning. Tankers are also not arriving on time.",
        "The tap water is coming completely muddy and dirty with foul smell. Children are getting sick.",
        "Sewage water has entered our ground floor compound due to choked drain line. Kindly arrange cleaning team urgently.",
        "Water pressure is very low in our sector {num}. We are unable to fill our overhead tanks.",
        "Manhole cover is broken and gutter is blocked near society main gate. It is very dangerous for two-wheelers.",
        "Contaminated water is being supplied through municipal pipeline for the past {days} days. Please inspect.",
        "Drainage line has burst in our colony road. Stagnant filthy water is breeding mosquitoes everywhere.",
        "We are receiving supply water only for 10 minutes that too at very odd hours around 2 AM.",
        "Gutter cleaning was promised last week by ward supervisor but nobody came to clear the choke.",
        "Water pump in public booster station is out of order since Monday. Entire block is suffering without water.",
        "Nallah near our colony is overflowing due to monsoon blockage. Please send suction machine immediately.",
        "Underground pipeline damaged by cable digging contractors, resulting in total water cut in our street."
    ],
    "Electricity & Power": [
        "There is frequent power supply interruption in our area. Please check the issue and resolve it.",
        "The street light in our area has not been working for the past week. It becomes very difficult to travel at night. Please fix it.",
        "Voltage fluctuation is very high in our colony. Our refrigerator and inverter have got damaged.",
        "Transformer near cross road #{num} is sparking heavily with loud buzzing sound. Please attend before fire catches.",
        "Sudden unscheduled load shedding for 6 hours daily without any prior SMS notification from electricity board.",
        "Low hanging electricity wires are touching tree branches in front of our house. It is a severe electrocution hazard.",
        "Electric pole got tilted after heavy rain and wire is hanging very low across the street.",
        "Meter reading recorded is abnormally high this month. Bill amount of Rs {amount} is triple our usual consumption.",
        "Power cut since 4 hours in scorching summer heat. Senior citizens and infants are struggling.",
        "Street lights remain ON throughout daytime wasting electricity while at night whole road is dark.",
        "Phase failure in our building line. Two phases are totally dead and water motors are not running.",
        "Feeder line tripped again for the third time today. Please depute technical lineman to inspect substation.",
        "Electric meter display has gone blank after thunderstorm. We are unable to track reading.",
        "DP box cover is open and live wires are exposed near children play area. Take immediate action.",
        "Frequent voltage drops are causing LED lights to flicker and AC compressor to shut down repeatedly."
    ],
    "Roads & Civic Infrastructure": [
        "The road near our society is damaged and has many potholes. Please repair it to avoid accidents.",
        "Main road connecting sector {num} is full of huge craters and uneven patches. Commuters are facing back pain and vehicle damage.",
        "Road dug up for pipeline work two months ago was left without proper tarring or resurfacing.",
        "Speed breaker near school junction has no paint or warning sign. Speeding vehicles are jumping dangerously.",
        "Footpath paving tiles are completely broken and encroached by hawkers, forcing pedestrians to walk on busy road.",
        "Deep ditch left open without barricades on bypass road. At night it is causing severe accident risks.",
        "Stormwater drain covers are missing on the roadside. Someone will fall inside during dark.",
        "Bitumen layer has washed away completely after first shower, exposing sharp gravel everywhere.",
        "Traffic signal at cross junction #{num} is not functioning since 3 days, causing heavy traffic jam during peak hours.",
        "Culvert bridge slab has developed structural cracks. Heavy trucks are still passing over it.",
        "Road widening work is halted halfway and loose stones and gravel are causing two-wheelers to skid daily.",
        "Open trench dug by telecom company has not been filled back since 20 days. Kindly arrange leveling.",
        "No zebra crossing or street signage installed near hospital entrance. Please mark it to prevent mishaps.",
        "Flyover service road has developed dangerous sinkhole. Please cordon off and repair urgently.",
        "Paver blocks on society approach road have sunk due to soil erosion, causing water logging."
    ],
    "Garbage & Sanitation": [
        "There is a lot of garbage near our building and it has not been collected for several days. Kindly arrange for cleaning.",
        "Garbage collection van (ghanta gadi) has not visited our street for the last 5 days. Waste is piling up.",
        "Community waste bin is overflowing on the footpath and stray animals are scattering garbage all over the road.",
        "Open dumping of hotel waste and plastic bags near public park causing unbearable foul odor and flies.",
        "Sanitation workers are dumping collected swept trash in open corner instead of lifting it to dump yard.",
        "Dead animal lying near roadside corner since morning. Please send sanitary disposal vehicle immediately.",
        "No door-to-door waste collection happening in our newly developed pocket in ward {num}.",
        "Public dustbins in market area are broken and broken glass and rotten food are scattered on walking path.",
        "Fumigation and mosquito fogging has not been done in our locality despite dengue cases rising.",
        "Debris and construction rubble dumped illegally on the pavement blocking pedestrian movement.",
        "Unsanitary conditions near public toilet complex due to lack of running water and daily sweeping.",
        "Kachra van driver refuses to collect segregated wet waste and argues with residents.",
        "Dry leaves and garden waste being burnt in open plot creating heavy toxic smoke in residential area.",
        "Open plot has turned into unauthorized garbage dump yard. Kindly clean and place a warning board.",
        "Green waste and tree cuttings dumped on roadside after cyclone have not been lifted for 2 weeks."
    ],
    "Public Transport": [
        "The bus did not arrive on time today, and there was no information about the delay. Please look into this.",
        "City bus route #{num} frequency is extremely poor during morning office hours. Buses are arriving overcrowded.",
        "Bus driver did not stop at designated bus shelter despite passengers waving their hands.",
        "Conductor was extremely rude and refused to return balance change of Rs {amount} to passengers.",
        "AC is not working in electric city bus #{num} and windows are sealed shut, suffocating passengers in peak traffic.",
        "Digital display boards at central bus terminal showing wrong arrival timings and wrong platform numbers.",
        "Last night bus on route {num} was cancelled without any prior announcement leaving office goers stranded.",
        "Bus tracking in the transport mobile app is showing GPS offline for all buses on our route.",
        "Drivers are driving rashly and overspeeding on city bypass, putting passenger lives in jeopardy.",
        "No wheelchair ramp or disabled assistance facility available at the city bus stand.",
        "Ticket vending machine at metro/bus interchange is out of order and token counter has 45 minutes queue.",
        "City bus broke down midway near flyover and no alternative bus was provided for over 1.5 hours.",
        "Auto-rickshaws outside railway station refusing to ply by meter and demanding exorbitant flat fares.",
        "Bus stop shelter glass roof is broken and passengers get drenched during heavy rains.",
        "Bus conductor refused to accept UPI QR code payment and forced passenger to deboard the vehicle."
    ],
    "Citizen & Consumer Services": [
        "I have submitted my application but have not received any update yet. Kindly check the status and provide an update.",
        "I received the wrong item in my order. Please arrange for a replacement or refund.",
        "The service is taking too long and my complaint has still not been resolved. Kindly provide a proper update.",
        "Applied for income / domicile certificate on portal {days} days back. Status is still showing 'Pending with Officer'.",
        "Ration card renewal application #{num} has been stuck in verification stage for over a month with no response.",
        "E-challan payment was deducted from my bank account via UPI but receipt was not generated on portal.",
        "Customer care helpline is never reachable. Call gets disconnected automatically after waiting 20 minutes.",
        "I ordered groceries online but half the items were missing from package and customer care is ignoring emails.",
        "Property tax payment made online is not reflecting in municipal ledger, and receipt shows unpaid.",
        "Name correction in Aadhaar/Voter ID was rejected without providing any valid reason or document clarification.",
        "Refund of Rs {amount} for cancelled ticket was promised within 3 working days but still not credited.",
        "Pension disbursement for senior citizen has been withheld for 2 months without any prior notice.",
        "Trade license renewal file has been delayed beyond government citizen charter guaranteed SLA.",
        "Consumer grievance ticket #{num} was marked closed by agent without resolving the actual issue.",
        "Delivery executive marked order as delivered with fake OTP entry while I was waiting at doorstep."
    ]
}

INDIAN_PREFIXES = [
    "",
    "Respected Sir/Madam, ",
    "Sir, ",
    "Hello authority, ",
    "URGENT ATTENTION: ",
    "To The Ward Officer, ",
    "Kind Attention Municipal Commissioner: ",
    "Regarding complaint: ",
    "Please do the needful: ",
    "I want to bring to your notice that ",
    "We are facing severe hardship because ",
    "This is to inform that "
]

INDIAN_SUFFIXES = [
    "",
    " Kindly do the needful at the earliest.",
    " Please look into this matter immediately.",
    " We request your prompt intervention to avoid inconvenience.",
    " Please take strict action and resolve it on top priority.",
    " Waiting for your swift resolution.",
    " Kindly send maintenance staff today itself.",
    " Otherwise residents will be forced to stage a dharna and approach consumer court.",
    " Please update ticket status as soon as possible.",
    " Hoping for immediate relief from the concerned department.",
    " Thanking you in anticipation."
]

def generate_indian_complaints(target_per_category=150):
    records = []
    
    for category, templates in COMPLAINT_TEMPLATES.items():
        for _ in range(target_per_category):
            template = random.choice(templates)
            num = random.randint(101, 999)
            days = random.randint(3, 21)
            amount = random.randint(50, 4500)
            
            body = template.format(num=num, days=days, amount=amount)
            prefix = random.choice(INDIAN_PREFIXES)
            suffix = random.choice(INDIAN_SUFFIXES)
            full_text = f"{prefix}{body}{suffix}".strip()
            
            records.append({
                "complaint_text": full_text,
                "department": category
            })
            
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def train_and_evaluate():
    print("Generating Indian citizen & consumer complaints dataset (~900 rows across 6 categories)...")
    df = generate_indian_complaints(target_per_category=150)
    print(f"Total complaints: {len(df)}")
    print(df["department"].value_counts())
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    data_path = os.path.join("data", "complaints_dataset.csv")
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to {data_path}")
    
    X = df["complaint_text"]
    y = df["department"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        max_features=5000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    classifier = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    classifier.fit(X_train_vec, y_train)
    
    y_pred = classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(vectorizer, os.path.join("models", "tfidf_vectorizer.joblib"))
    joblib.dump(classifier, os.path.join("models", "complaint_classifier.joblib"))
    print("Models saved successfully to ./models/")

if __name__ == "__main__":
    train_and_evaluate()
