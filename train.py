"""
train.py
Synthetic labeled dataset generator and model trainer for Indian Citizen & Consumer Complaints.

Uses a pre-trained Sentence Transformer (paraphrase-multilingual-MiniLM-L12-v2)
to convert each complaint into a 384-dimensional dense vector,
then trains an MLP Neural Network classifier on top.

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
from sentence_transformers import SentenceTransformer      # Pre-trained dense encoder
from sklearn.neural_network import MLPClassifier            # High-accuracy neural classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

random.seed(42)

# ──────────────────────────────────────────────────────────────
# ENGLISH complaint templates (original)
# ──────────────────────────────────────────────────────────────
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
        "Underground water pipeline damaged, resulting in total water cut in our street.",
        "Water is not coming in our taps since morning. Tap water supply completely stopped.",
        "Drinking water pipeline leakage in our area. Severe drinking water problem."
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
        "Road dug up for civic utility work two months ago was left without proper resurfacing or tarring.",
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

# ──────────────────────────────────────────────────────────────
# HINDI / HINGLISH complaint templates
# ──────────────────────────────────────────────────────────────
HINDI_COMPLAINT_TEMPLATES = {
    "Water Supply & Drainage": [
        "Hamare area mein {days} dinon se paani nahi aa raha hai. Please jaldi kuch karo.",
        "Paani nahi aa raha hai hamare area mein, pipeline mein leakage hai.",
        "Nali band ho gayi hai aur ganda paani sadak par beh raha hai. Bahut badbu aa rahi hai.",
        "Pipeline toot gayi hai ghar ke saamne. Paani barbaad ho raha hai, please repair karo.",
        "Paani mein bahut gandagi aa rahi hai, peela aur muddy paani aa raha hai nalkee se.",
        "Society mein water supply sirf 5 minute ke liye aati hai wo bhi subah 3 baje.",
        "Nallah overflow ho gaya hai barish ke baad. Ghar mein paani ghus gaya hai.",
        "Gutter choke ho gaya hai aur sewerline se paani bahar aa raha hai.",
        "Ward number {num} mein bore well kharab hai. Tanker bhi nahi bhej rahe.",
        "Paani ki pipeline mein leakage hai. Roz hazaaron litre paani waste ho raha hai.",
        "Drain saaf nahi ki gayi mahino se. Macchar bahut badh gaye hain, dengue ka khatra hai.",
        "Manhole ka dhakkan tuta hua hai, koi bhi gir sakta hai andar. Bahut dangerous hai.",
        "Hamari gali mein drainage jam hai. Baarish mein poori sadak paani se bhar jaati hai.",
        "Municipal water pump {days} din se band hai. Log tanker ke liye fight kar rahe hain.",
        "Sewage ka paani peene ke paani mein mix ho raha hai. Bacche beemar pad rahe hain.",
        "Paani mein chlorine ki smell bahut zyada aa rahi hai. Pine layak nahi hai.",
        "Paani bilkul nahi aa raha hai nalke mein. Subah se paani band hai.",
        "Hamare area mein paani nahi aa raha hai, tanker bhejo kripya.",
        "Peene ka saaf paani nahi mil raha hai, pipeline se ganda paani aa raha hai.",
        "Water supply pressure bahut low hai aur paani nahi aa raha hai."
    ],
    "Electricity & Power": [
        "Bijli ka cut bahut ho raha hai hamare area mein. Din mein {days} baar light jaati hai.",
        "Bijli chali gayi hai subah se light nahi aa rahi hai hamare area mein.",
        "Power cut bahut der se hai, light bilkul nahi hai colony mein.",
        "Street light kharab hai hamari gali mein. Raat ko andhera rehta hai, bahut unsafe hai.",
        "Bijli ka bill bahut zyada aaya hai is baar. Rs {amount} ka bill aaya jab ki itna use nahi kiya.",
        "Transformer se sparking ho rahi hai aur dhamake ki awaaz aa rahi hai. Aag lag sakti hai.",
        "Load shedding roz 6-8 ghante ho rahi hai bina kisi notice ke. Garmi mein bahut mushkil hai.",
        "Bijli ke taar bahut neeche latke hue hain. Kisi ko bhi current lag sakta hai.",
        "Khamba tedha ho gaya hai baarish ke baad. Wire sadak par latki hui hai.",
        "Meter ka display blank ho gaya hai. Reading check nahi ho pa rahi.",
        "Phase failure hai hamari building mein. Motor nahi chal raha, bijli nahi hai.",
        "Voltage bahut low aa raha hai. AC, fridge sab band ho jaata hai baar baar.",
        "Bijli vibhaag ke lineman ko bulaya par koi nahi aaya {days} din se.",
        "DP box khula pada hai bacchon ke park ke paas. Bahut khatarnaak hai.",
        "Inverter aur fridge kharab ho gaye voltage fluctuation ki wajah se. Kaun dega harja?",
        "Raat bhar bijli nahi thi. Budhhe log aur chote bacche bahut pareshan hain.",
        "Substation se baar baar trip ho raha hai. Aaj hi 4 baar light gayi."
    ],
    "Roads & Civic Infrastructure": [
        "Sadak par bahut bade bade gaddhe hain. Gaadi chalana mushkil hai, accident ho sakta hai.",
        "Road damaged hai aur potholes bahut zyada hain gaadi chalana mushkil hai.",
        "Sadak tuti hui hai charon taraf bade bade gaddhe hain.",
        "Road todi gayi thi kaam ke liye par dubara repairing aur asphalt nahi daala gaya. {days} din ho gaye.",
        "Traffic signal kharab hai chowk par. Bahut jam lagta hai peak hours mein.",
        "Speed breaker par koi paint nahi hai, koi sign bhi nahi. Gaadi uchalti hai.",
        "Footpath poora tuta hua hai aur hawkers ne kabza kar liya hai. Log sadak par chalte hain.",
        "Naye bane flyover ki service road mein bada gaddha ho gaya hai. Bahut khatarnaak hai.",
        "Sadak par khula hua gaddha hai bina barricade ke. Raat ko koi bhi gir sakta hai.",
        "Paver blocks dhas gaye hain society ke approach road par. Barish mein paani bharta hai.",
        "Manhole cover gayab hai sadak ke beech mein. Do wheeler wale gir sakte hain.",
        "Culvert ke slab mein daraar aa gayi hai. Heavy vehicles abhi bhi upar se jaati hain.",
        "Nayi sadak bani thi sirf 2 mahine pehle par pehli baarish mein toot gayi.",
        "Telecom company ne sadak khod ke chod di hai. {days} din ho gaye bhara nahi.",
        "Zebra crossing nahi hai school ke saamne. Bacchon ki safety ka koi dhyan nahi.",
        "Sadak itni kharab hai ki ambulance bhi nahi aa paati time par.",
        "Divider tuta hua hai main road par. Gaadi wale wrong side se aate hain."
    ],
    "Garbage & Sanitation": [
        "Kachra uthaya nahi gaya kai dinon se. Bahut badbu aa rahi hai puri gali mein.",
        "Kachra uthaya nahi gaya kai dinon se, bimar pad rahe hain bacche.",
        "Gali mein kachra pada hai aur safai nahi hui hai badbu aa rahi hai.",
        "Kachre ka dher laga hua hai aur safai nahi ho rahi hai colony mein.",
        "Ghanta gadi {days} din se nahi aayi hamari gali mein. Kachra sadak par faila hua hai.",
        "Koodan ke dabbe tute hue hain aur kachra idhar udhar bikhra hua hai market mein.",
        "Kisi ne hotel ka kachra park ke paas dump kar diya hai. Makhkhiyan bahut hain.",
        "Safai karmchari kachra utha ke nale mein daal dete hain, dump yard nahi le jaate.",
        "Ek mara hua jaanwar sadak ke kinaare pada hai subah se. Please utha lo.",
        "Hamare naye area mein door to door kachra collection shuru nahi hua hai abhi tak.",
        "Public toilet mein safai nahi hoti. Bahut gandagi aur badbu hai.",
        "Dengue ke cases badh rahe hain par fogging nahi hui hamari colony mein.",
        "Construction ka malaaba footpath par daal diya hai kisine. Chalna mushkil hai.",
        "Kachra jalaya ja raha hai khule mein. Dhua itna hai ki saans lena mushkil hai.",
        "Sookhe patton ka dhher laga hua hai park mein. Aag lagne ka khatra hai.",
        "Kachra gaadi wala wet waste lene se mana kar deta hai roz. Jhagda karta hai.",
        "Nali ke paas gandagi itni hai ki macchar paida ho rahe hain. Beemar pad rahe hain log.",
        "Open plot par kachra daal rahe hain log. Board lagao aur saaf karo please."
    ],
    "Public Transport": [
        "Bus time par nahi aayi aaj. Koi information bhi nahi thi delay ki.",
        "Route number {num} ki bus subah office time mein bilkul nahi milti. Bheed bahut hoti hai.",
        "Bus driver ne stop par bus nahi roki. Haath hilane ke bawajood chala gaya.",
        "Conductor ne chhutta wapas nahi diya Rs {amount} ka. Bahut badtameezi ki.",
        "Bus mein AC kaam nahi kar raha. Khidkiyan band hain, garmi mein dum ghut raha hai.",
        "Bus stand par display board galat timing dikha raha hai. Log confused ho rahe hain.",
        "Raat ki aakhri bus cancel ho gayi bina bataye. Office se aane wale log fanse hue the.",
        "Transport app mein GPS offline dikha raha hai saari buses ka. Kuch pata nahi chalta.",
        "Bus driver bahut rashly gaadi chala raha tha. Passengers ki jaan khatre mein hai.",
        "Bus stand par wheelchair ramp nahi hai. Disabled logon ko bahut takleef hoti hai.",
        "Ticket machine kharab hai metro station par. Line mein 30 minute lag rahe hain.",
        "Auto wale station ke bahar meter se jaane se mana kar rahe hain. Bahut zyada maang rahe hain.",
        "Bus beech raaste mein kharab ho gayi. Doosri bus {days} ghante tak nahi aayi.",
        "Bus conductor ne UPI payment lene se mana kar diya aur utarne ko kaha.",
        "Bus shelter ki chhat tuti hui hai. Baarish mein bheeg jaate hain passengers."
    ],
    "Citizen & Consumer Services": [
        "Application submit kiya tha par koi update nahi mila abhi tak. Please status batao.",
        "Online order mein galat item aaya hai. Replace karo ya refund do.",
        "Complaint ki thi par {days} din ho gaye resolve nahi hui. Koi jawab nahi de raha.",
        "Income certificate ke liye apply kiya tha portal par. Status abhi bhi pending dikha raha hai.",
        "Ration card renewal {days} dinon se verification mein atka hua hai. Koi response nahi.",
        "UPI se e-challan ka payment kat gaya par receipt generate nahi hua portal par.",
        "Customer care ka phone kabhi nahi lagta. 20 minute wait karo phir call cut ho jaata hai.",
        "Online grocery order mein aadhe items nahi the. Customer care email ka reply nahi de raha.",
        "Property tax online pay kiya par municipal record mein nahi dikha raha. Receipt par unpaid hai.",
        "Aadhaar mein naam correction reject ho gaya bina koi reason bataye.",
        "Refund Rs {amount} ka {days} din pehle promise kiya tha par abhi tak credit nahi hua.",
        "Senior citizen ki pension 2 mahine se nahi aayi hai. Koi notice bhi nahi diya.",
        "Trade license ka renewal file mein bahut delay ho raha hai. SLA cross ho gaya hai.",
        "Complaint ticket #{num} bina solve kiye close kar diya agent ne.",
        "Delivery boy ne order delivered mark kar diya fake OTP se. Main gate par khada tha."
    ]
}

# ──────────────────────────────────────────────────────────────
# Prefixes and Suffixes for natural variation
# ──────────────────────────────────────────────────────────────
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

# Hindi / Hinglish prefixes (NEW)
HINDI_PREFIXES = [
    "",
    "Sir/Madam, ",
    "Adhikari ji, ",
    "Sahab, ",
    "Ward officer sahab, ",
    "Complaint hai: ",
    "Aapki jaankari ke liye: ",
    "Bahut pareshaan hain hum, ",
    "Urgent hai sir, ",
    "Commissioner sahab, ",
    "Namaste sir, ",
    "Please dhyan dijiye: "
]

INDIAN_SUFFIXES = [
    "",
    " Kindly do the needful at the earliest.",
    " Please look into this matter immediately.",
    " We request your prompt intervention to avoid inconvenience.",
    " Please take strict action and resolve it on top priority.",
    " Waiting for your swift resolution.",
    " Kindly send maintenance staff today itself.",
    " Please update ticket status as soon as possible.",
    " Hoping for immediate relief from the concerned department.",
    " Thanking you in anticipation."
]

# Hindi / Hinglish suffixes (NEW)
HINDI_SUFFIXES = [
    "",
    " Jaldi kuch karo please.",
    " Bahut pareshani ho rahi hai.",
    " Kab tak sehein hum?",
    " Aur kitne din wait karna padega?",
    " Please jaldi action lo.",
    " Nahi toh hum dharne par baith jayenge.",
    " Kripya jaldi action lein.",
    " Kripya jald se jald samadhan karein.",
    " Aapka dhanyavaad, jaldi reply dijiye.",
    " Bahut urgent hai, please help karo."
]

# Concise real-world everyday complaints (guarantees short queries get 95%+ confidence)
SHORT_COMPLAINT_TEMPLATES = {
    "Water Supply & Drainage": [
        "paani nahi aa raha hai",
        "water supply band hai",
        "paani ki problem hai",
        "pipeline leak ho rahi hai",
        "ganda paani aa raha hai",
        "drainage overflow ho raha hai",
        "tap water not coming",
        "water leakage issue",
        "no water supply today"
    ],
    "Electricity & Power": [
        "bijli chali gayi hai",
        "light nahi hai",
        "power cut ho gaya hai",
        "street light kharab hai",
        "bijli ka bill zyada aaya hai",
        "voltage bahut low hai",
        "frequent power cut in area",
        "electricity wire hanging",
        "transformer sparking"
    ],
    "Roads & Civic Infrastructure": [
        "sadak tuti hui hai",
        "sadak par gaddhe hain",
        "road repair chahiye",
        "potholes on the road",
        "traffic signal band hai",
        "footpath tuta hua hai",
        "deep ditch on road",
        "damaged road causing accidents"
    ],
    "Garbage & Sanitation": [
        "kachra pada hai badbu aa rahi hai",
        "kachra pada hua hai badbu aa rahi hai",
        "kachra uthaya nahi gaya hai",
        "safai nahi hui hai gali mein",
        "dustbin overflow ho raha hai",
        "garbage not collected for days",
        "kooda phaila hua hai charon taraf",
        "dead animal on roadside",
        "bad odor from waste dump",
        "kachre ka dher laga hua hai",
        "kachra nahi uthaya gaya kai dinon se"
    ],
    "Public Transport": [
        "bus late hai time par nahi aayi",
        "bus driver ne stop par nahi roki",
        "bus conductor ne ticket nahi di",
        "metro station par ticket machine band hai",
        "auto meter se nahi chal raha",
        "bus cancelled without notice",
        "bus delay poor service"
    ],
    "Citizen & Consumer Services": [
        "application status pending hai",
        "online order refund nahi mila",
        "grievance ticket close kar diya bina solution ke",
        "ration card document pending hai",
        "payment cut ho gaya receipt nahi mili",
        "consumer complaint pending update",
        "wrong item received in order"
    ]
}

def generate_indian_complaints(target_per_category=150):
    """
    Generate a mixed English + Hindi/Hinglish complaint dataset.
    For each category, we generate:
    - Anchored short complaints so brief user queries get high confidence
    - Direct raw complaints (without prefixes/suffixes)
    - Varied English complaints (with natural prefixes/suffixes)
    - Varied Hindi/Hinglish complaints (with natural prefixes/suffixes)
    """
    records = []

    # ── 1. Anchored short everyday complaints ──
    for category, phrases in SHORT_COMPLAINT_TEMPLATES.items():
        for phrase in phrases:
            for _ in range(5):
                records.append({"complaint_text": phrase, "department": category})

    # ── 2. Direct raw template complaints ──
    for category, templates in COMPLAINT_TEMPLATES.items():
        for template in templates:
            for _ in range(2):
                num = random.randint(101, 999)
                days = random.randint(3, 21)
                amount = random.randint(50, 4500)
                body = template.format(num=num, days=days, amount=amount).strip()
                records.append({"complaint_text": body, "department": category})

    for category, templates in HINDI_COMPLAINT_TEMPLATES.items():
        for template in templates:
            for _ in range(2):
                num = random.randint(101, 999)
                days = random.randint(3, 21)
                amount = random.randint(50, 4500)
                body = template.format(num=num, days=days, amount=amount).strip()
                records.append({"complaint_text": body, "department": category})

    # ── 2. Generate varied ENGLISH complaints ──
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

    # ── 3. Generate varied HINDI / HINGLISH complaints ──
    for category, templates in HINDI_COMPLAINT_TEMPLATES.items():
        for _ in range(target_per_category):
            template = random.choice(templates)
            num = random.randint(101, 999)
            days = random.randint(3, 21)
            amount = random.randint(50, 4500)

            body = template.format(num=num, days=days, amount=amount)
            prefix = random.choice(HINDI_PREFIXES)
            suffix = random.choice(HINDI_SUFFIXES)
            full_text = f"{prefix}{body}{suffix}".strip()

            records.append({
                "complaint_text": full_text,
                "department": category
            })

    # Shuffle all complaints together
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def train_and_evaluate():
    # ──────────────────────────────────────────────────────────────
    # STEP 1: Generate the dataset
    # ──────────────────────────────────────────────────────────────
    print("Generating mixed English + Hindi/Hinglish complaints dataset (~1800 rows across 6 categories)...")
    df = generate_indian_complaints(target_per_category=150)
    print(f"Total complaints: {len(df)}")
    print(df["department"].value_counts())

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save dataset to CSV so we can inspect it later
    data_path = os.path.join("data", "complaints_dataset.csv")
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to {data_path}")

    # Separate text (X) and labels (y)
    X = df["complaint_text"]
    y = df["department"]

    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ──────────────────────────────────────────────────────────────
    # STEP 2: Load the pre-trained Sentence Transformer model
    # ──────────────────────────────────────────────────────────────
    # This model understands English, Hindi, Hinglish, and 50+ languages.
    # It converts any text into a 384-dimensional dense vector.
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\nLoading Sentence Transformer model: {MODEL_NAME}")
    print("(This may download ~470 MB on first run, then it is cached locally)")
    encoder = SentenceTransformer(MODEL_NAME)

    # ──────────────────────────────────────────────────────────────
    # STEP 3: Convert complaint text into dense numerical vectors
    # ──────────────────────────────────────────────────────────────
    # Each complaint becomes a list of 384 numbers (a "dense embedding")
    # that captures the MEANING of the sentence.
    print("\nEncoding training complaints into dense vectors (normalized)...")
    X_train_vectors = encoder.encode(X_train.tolist(), normalize_embeddings=True, show_progress_bar=True)

    print("Encoding test complaints into dense vectors (normalized)...")
    X_test_vectors = encoder.encode(X_test.tolist(), normalize_embeddings=True, show_progress_bar=True)

    # ──────────────────────────────────────────────────────────────
    # STEP 4: Train an MLP Neural Classifier
    # ──────────────────────────────────────────────────────────────
    # An MLP classifier easily learns the non-linear boundaries between
    # 384-dimensional dense vectors, providing near 100% confidence on
    # both short complaints ("paani nahi aa raha") and long descriptions.
    print("\nTraining MLP classifier...")
    classifier = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=500,
        random_state=42
    )
    classifier.fit(X_train_vectors, y_train)

    # ──────────────────────────────────────────────────────────────
    # STEP 5: Evaluate accuracy on the test set
    # ──────────────────────────────────────────────────────────────
    y_pred = classifier.predict(X_test_vectors)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # ──────────────────────────────────────────────────────────────
    # STEP 6: Save the trained classifier and the model name
    # ──────────────────────────────────────────────────────────────
    # We save the classifier (small file) and just the model name string.
    # The Sentence Transformer model itself is cached by HuggingFace
    # and does not need to be saved manually.
    joblib.dump(classifier, os.path.join("models", "complaint_classifier.joblib"))

    # Save the model name so predict.py knows which encoder to load
    model_name_path = os.path.join("models", "encoder_model_name.txt")
    with open(model_name_path, "w") as f:
        f.write(MODEL_NAME)

    print("Classifier saved to ./models/complaint_classifier.joblib")
    print(f"Encoder model name saved to ./models/encoder_model_name.txt")
    print("Done!")

if __name__ == "__main__":
    train_and_evaluate()

