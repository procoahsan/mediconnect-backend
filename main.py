# Refactored imports
import re
import torch
import jwt
import pymongo
import pandas as pd
import numpy as np
from jwt.exceptions import InvalidTokenError
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.encoders import jsonable_encoder
from passlib.context import CryptContext
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import google.generativeai as genai

# Initialize app and CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT and auth config
SECRET_KEY = "hello"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# MongoDB config
client = pymongo.MongoClient("mongodb+srv://usamaejaz274:usama876559@cluster0.mklspwo.mongodb.net/")
db = client["FYP"]
userCol = db["users"]
appointmentCol = db["appointments"]

# Suggested doctor mapping with times
DOCTOR_DISEASE_MAP = [
    {"doctor": "Dr. Sam", "disease": "fever", "available_times": ["10:00 AM", "2:00 PM", "4:30 PM"]},
    {"doctor": "Dr. Paul", "disease": "pneumonia", "available_times": ["9:30 AM", "1:00 PM", "3:45 PM"]},
    {"doctor": "Dr. Clara", "disease": "diabetes", "available_times": ["11:00 AM", "3:00 PM", "6:00 PM"]},
    {"doctor": "Dr. Ahmed", "disease": "hypertension", "available_times": ["10:15 AM", "2:30 PM", "5:00 PM"]},
    {"doctor": "Dr. Meera", "disease": "asthma", "available_times": ["9:00 AM", "12:00 PM", "4:00 PM"]},
    {"doctor": "Dr. John", "disease": "migraine", "available_times": ["11:30 AM", "1:30 PM", "5:15 PM"]},
    {"doctor": "Dr. Nina", "disease": "allergy", "available_times": ["8:45 AM", "12:30 PM", "3:30 PM"]},
    {"doctor": "Dr. Rakesh", "disease": "arthritis", "available_times": ["10:00 AM", "1:15 PM", "4:45 PM"]},
    {"doctor": "Dr. Emily", "disease": "anemia", "available_times": ["9:15 AM", "11:45 AM", "2:45 PM"]},
    {"doctor": "Dr. Rahul", "disease": "thyroid", "available_times": ["8:30 AM", "12:15 PM", "3:00 PM"]}
]

# Models and tokenizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    tokenizer = AutoTokenizer.from_pretrained("model")
    tokenizer.pad_token = tokenizer.eos_token
    config = PeftConfig.from_pretrained("model")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to(device)
    model = PeftModel.from_pretrained(base_model, "model").to(device)
    model.eval()
except Exception as e:
    print("Model loading error:", str(e))
    tokenizer = model = None

# Load and preprocess dataset
data = pd.read_csv("appointment_scheduling_dataset.csv")
data["appointment_datetime"] = pd.to_datetime(data["appointment_date"] + " " + data["appointment_time"])
data.sort_values("appointment_datetime", inplace=True)
label_encoders = {}
for col in ["doctor_name", "specialization", "status"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

data["day_of_week"] = data["appointment_datetime"].dt.dayofweek
data["hour"] = data["appointment_datetime"].dt.hour
data["minute"] = data["appointment_datetime"].dt.minute
features = ["doctor_name", "specialization", "day_of_week", "hour", "minute"]
X, y = data[features], data["appointment_datetime"].astype("int64") // 10 ** 9
model_rf = RandomForestRegressor(n_estimators=30).fit(X, y)

# Pydantic Models
class User(BaseModel):
    username: str
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    source: Optional[str] = "medical_bot"

class AppointmentRequest(BaseModel):
    doctor_name: str
    patient_name: str
    age: int
    gender: str
    chosen_slot: str

# Auth utils
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401)
        user = userCol.find_one({"email": email}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401)
        return user
    except InvalidTokenError:
        raise HTTPException(status_code=401)

# Doctor suggestion logic
def append_suggested_doctor(user_query: str, bot_response: str) -> str:
    combined_text = f"{user_query.lower()} {bot_response.lower()}"
    for mapping in DOCTOR_DISEASE_MAP:
        if mapping['disease'].lower() in combined_text:
            times = ", ".join(mapping["available_times"])
            return f"{bot_response}\n\nSuggested Doctor: {mapping['doctor']}\nAvailable Times: {times}"
    return bot_response

# Chat with model
def chat(input_text: str) -> str:
    if tokenizer is None or model is None:
        return "Error: Bot not available."
    prompt = (
        "You should act as a highly trained medical professional. Please provide specific health advice "
        f"related to the patient's query.\nPatient: {input_text}\nDoctor: "
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2, top_p=0.9,
                                 do_sample=True, repetition_penalty=1.2, early_stopping=True,
                                 pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Doctor:")[-1].strip()

# Gemini AI response
def get_gemini_medical_response(user_input: str) -> str:
    try:
        genai.configure(api_key="AIzaSyDoG89hsCX3CN3W3nRJz3ZllA_zq9zSJgw")
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(
            f"You are a medical assistant. Provide general advice only.\nQuery: {user_input}")
        return response.text if response and response.text else "No response"
    except Exception as e:
        return f"Gemini error: {str(e)}"

# Slot prediction logic
def get_next_available_slots(doctor_name: str, n=10, interval=15):
    doc_id = label_encoders['doctor_name'].transform([doctor_name])[0]
    doc_data = data[data['doctor_name'] == doc_id]
    latest = max(datetime.now(), doc_data['appointment_datetime'].max())
    slots = []
    for _ in range(n):
        feats = [[doc_id, doc_data['specialization'].iloc[0], latest.weekday(), latest.hour, latest.minute]]
        pred_time = datetime.fromtimestamp(model_rf.predict(feats)[0])
        pred_time = max(pred_time, latest + timedelta(minutes=interval))
        while pred_time in doc_data['appointment_datetime'].values:
            pred_time += timedelta(minutes=interval)
        slots.append(pred_time.strftime('%Y-%m-%d %I:%M %p'))
        latest = pred_time + timedelta(minutes=interval)
    return slots

# Booking logic
def allocate_appointment(doctor_name, patient_name, age, gender, slot, email):
    global data
    if doctor_name not in label_encoders['doctor_name'].classes_:
        raise HTTPException(status_code=400, detail="Invalid doctor")
    doc_id = label_encoders['doctor_name'].transform([doctor_name])[0]
    spec_id = data.loc[data['doctor_name'] == doc_id, 'specialization'].iloc[0]
    room = int(re.search(r'\d+', data.loc[data['doctor_name'] == doc_id, 'room_number'].iloc[0]).group())
    dt = pd.to_datetime(slot, format='%Y-%m-%d %I:%M %p')
    record_df = pd.DataFrame([{
        'appointment_id': int(data['appointment_id'].max() + 1),
        'doctor_name': doc_id,
        'specialization': spec_id,
        'room_number': room,
        'patient_name': patient_name,
        'age': age,
        'gender': gender,
        'appointment_date': slot.split()[0],
        'appointment_time': ' '.join(slot.split()[1:]),
        'status': label_encoders['status'].transform(['Scheduled'])[0],
        'appointment_datetime': dt
    }])
    data = pd.concat([data, record_df], ignore_index=True)
    mongo_doc = {
        'appointment_id': int(record_df['appointment_id'].iloc[0]),
        'doctor_name': doctor_name,
        'specialization': label_encoders['specialization'].inverse_transform([spec_id])[0],
        'room_number': room,
        'patient_name': patient_name,
        'age': age,
        'gender': gender,
        'appointment_date': slot.split()[0],
        'appointment_time': ' '.join(slot.split()[1:]),
        'status': 'Scheduled',
        'user_email': email,
        'created_at': datetime.utcnow()
    }
    appointmentCol.insert_one(mongo_doc.copy())
    return jsonable_encoder(mongo_doc)

# Routes
@app.get("/api/health")
async def health():
    return {"status": "API is running"}

@app.post("/api/chat")
async def chat_api(chat_request: ChatRequest):
    msg = chat_request.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message required")
    if chat_request.source.lower() == "gemini":
        res = get_gemini_medical_response(msg)
        source = "gemini"
    else:
        res = chat(msg)
        source = "medical_bot"
    full_response = append_suggested_doctor(msg, res)
    return {"status": "success", "response": full_response, "source": source}

@app.post("/signup")
async def signup(user: User):
    if userCol.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="User exists")
    userCol.insert_one({"username": user.username, "email": user.email, "password": get_password_hash(user.password)})
    return {"message": "User created"}

@app.post("/login")
async def login(login_data: Login):
    user = userCol.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token({"sub": login_data.email})}

@app.get("/api/doctors")
async def get_doctors(current_user: dict = Depends(get_current_user)):
    try:
        return {"doctors": sorted(set(label_encoders['doctor_name'].inverse_transform(data['doctor_name']))) }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available-slots")
async def slots(doctor_name: str, current_user: dict = Depends(get_current_user)):
    if not doctor_name:
        raise HTTPException(status_code=400, detail="Doctor name required")
    return {"slots": get_next_available_slots(doctor_name)}

@app.post("/api/appointments")
async def book(appointment: AppointmentRequest, current_user: dict = Depends(get_current_user)):
    return {
        "message": "Appointment scheduled",
        "appointment": allocate_appointment(
            appointment.doctor_name, appointment.patient_name,
            appointment.age, appointment.gender,
            appointment.chosen_slot, current_user['email']
        )
    }

@app.get("/api/appointments")
async def appointments(current_user: dict = Depends(get_current_user)):
    return {"appointments": jsonable_encoder(list(appointmentCol.find({"user_email": current_user['email']}, {"_id": 0})))}
