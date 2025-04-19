import re
from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import pymongo
import jwt  # PyJWT library
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from typing import Optional
import google.generativeai as genai

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = "hello"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# MongoDB Configuration
client = pymongo.MongoClient('mongodb+srv://usamaejaz274:usama876559@cluster0.mklspwo.mongodb.net/')
db = client['FYP']
userCol = db['users']
appointmentCol = db['appointments']

# Medical Chat Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained("model")
    tokenizer.pad_token = tokenizer.eos_token
    config = PeftConfig.from_pretrained("model")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to(device)
    model = PeftModel.from_pretrained(base_model, "model").to(device)
    model.eval()
    print("Tokenizer and model loaded successfully")
except Exception as e:
    print(f"Error loading medical chat model: {str(e)}")
    tokenizer = None
    model = None

# Load and preprocess appointment scheduling data
data = pd.read_csv('appointment_scheduling_dataset.csv')
data['appointment_datetime'] = pd.to_datetime(data['appointment_date'] + ' ' + data['appointment_time'])
data.sort_values('appointment_datetime', inplace=True)

label_encoders = {}
for col in ['doctor_name', 'specialization', 'status']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

data['day_of_week'] = data['appointment_datetime'].dt.dayofweek
data['hour'] = data['appointment_datetime'].dt.hour
data['minute'] = data['appointment_datetime'].dt.minute

features = ['doctor_name', 'specialization', 'day_of_week', 'hour', 'minute']
X = data[features]
y = data['appointment_datetime'].astype('int64') // 10 ** 9

model_rf = RandomForestRegressor(n_estimators=30)
model_rf.fit(X, y)

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

# JWT Helper Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception
    user = userCol.find_one({"email": email}, {'_id': 0})  # Exclude _id
    if user is None:
        raise credentials_exception
    return user

# Medical Chat Function
def chat(patient_input: str, max_new_tokens: int = 100):
    if tokenizer is None or model is None:
        return "Error: Medical bot is not properly configured."

    prompt = (
        "You should act as a highly trained medical professional. Please provide specific health advice "
        "related to the patient's query. Focus on medical accuracy and professionalism. \n"
        f"Patient: {patient_input}\nDoctor: "
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    doctor_response = response.split("Doctor:")[-1].strip()
    return doctor_response

# Gemini Placeholder Function
def get_gemini_medical_response(user_input: str) -> str:
    try:
        genai.configure(api_key="AIzaSyDoG89hsCX3CN3W3nRJz3ZllA_zq9zSJgw")
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a medical information assistant. Provide accurate and general information "
            "related to medical queries. Do not provide personal medical advice or diagnoses. "
            f"Respond to the following query: {user_input}"
        )
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            return response.text
        else:
            return "Error: No valid response from Gemini"
    except Exception as e:
        return f"Gemini API error: {str(e)}"

# Helper Functions for Appointment Scheduling
def get_next_available_slots(doctor_name: str, n_slots: int = 10, interval_minutes: int = 15):
    doctor_id = label_encoders['doctor_name'].transform([doctor_name])[0]
    doctor_data = data[data['doctor_name'] == doctor_id]
    latest_time = max(datetime.now(), doctor_data['appointment_datetime'].max())

    next_slots = []
    for _ in range(n_slots):
        new_features = [[doctor_id, doctor_data['specialization'].iloc[0],
                        latest_time.weekday(), latest_time.hour, latest_time.minute]]
        predicted_timestamp = model_rf.predict(new_features)[0]
        predicted_time = datetime.fromtimestamp(predicted_timestamp)
        predicted_time = max(predicted_time, latest_time + timedelta(minutes=interval_minutes))

        while predicted_time in doctor_data['appointment_datetime'].values:
            predicted_time += timedelta(minutes=interval_minutes)

        next_slots.append(predicted_time)
        latest_time = predicted_time + timedelta(minutes=interval_minutes)

    return [slot.strftime('%Y-%m-%d %I:%M %p') for slot in next_slots]

def allocate_appointment(doctor_name: str, patient_name: str, age: int, gender: str, chosen_slot: str, user_email: str):
    global data
    # Validate doctor_name
    if doctor_name not in label_encoders['doctor_name'].classes_:
        raise HTTPException(status_code=400, detail="Invalid doctor name")

    doctor_id = label_encoders['doctor_name'].transform([doctor_name])[0]
    room_str = data.loc[data['doctor_name'] == doctor_id, 'room_number'].iloc[0]
    room_number = int(re.search(r'\d+', room_str).group())

    # Validate chosen_slot format
    try:
        pd.to_datetime(chosen_slot, format='%Y-%m-%d %I:%M %p')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format for chosen_slot. Use 'YYYY-MM-DD HH:MM AM/PM'")

    # Prepare appointment for DataFrame (with encoded values)
    new_appointment_df = {
        'appointment_id': int(data['appointment_id'].max() + 1),
        'doctor_name': doctor_id,
        'specialization': int(data.loc[data['doctor_name'] == doctor_id, 'specialization'].iloc[0]),
        'room_number': room_number,
        'patient_name': patient_name,
        'age': int(age),
        'gender': gender,
        'appointment_date': chosen_slot.split()[0],
        'appointment_time': ' '.join(chosen_slot.split()[1:]),
        'status': label_encoders['status'].transform(['Scheduled'])[0],
        'appointment_datetime': pd.to_datetime(chosen_slot)
    }

    # Convert to DataFrame and concatenate
    new_appointment_df = pd.DataFrame([new_appointment_df])
    data = pd.concat([data, new_appointment_df], ignore_index=True)

    # Prepare appointment for MongoDB (with human-readable values and Python-native types)
    new_appointment_mongo = {
        'appointment_id': int(new_appointment_df['appointment_id'].iloc[0]),  # Ensure Python int
        'doctor_name': doctor_name,
        'specialization': label_encoders['specialization'].inverse_transform(
            [int(data.loc[data['doctor_name'] == doctor_id, 'specialization'].iloc[0])])[0],
        'room_number': room_number,
        'patient_name': patient_name,
        'age': int(age),
        'gender': gender,
        'appointment_date': chosen_slot.split()[0],
        'appointment_time': ' '.join(chosen_slot.split()[1:]),
        'status': 'Scheduled',
        'user_email': user_email,
        'created_at': datetime.utcnow()
    }

    # Save to MongoDB
    appointmentCol.insert_one(new_appointment_mongo.copy())  # Use copy to avoid modifying original

    # Return serialized document to ensure JSON compatibility
    return jsonable_encoder(new_appointment_mongo)

# API Endpoints
@app.get("/api/health")
async def health_check():
    return {"status": "API is running"}

@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_message = chat_request.message
    source = chat_request.source.lower()

    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Invalid message")

    if source == "gemini":
        response = get_gemini_medical_response(user_message)
        source_used = "gemini"
    else:
        response = chat(user_message)
        source_used = "medical_bot"

    return {
        "status": "success",
        "response": response,
        "user_message": user_message,
        "source": source_used
    }

@app.post("/signup")
async def signup(user: User):
    existing_user = userCol.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = get_password_hash(user.password)
    new_user = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password
    }
    userCol.insert_one(new_user)
    return {"message": "User created successfully"}

@app.post("/login")
async def login(login_data: Login):
    user = userCol.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": login_data.email})
    return {"access_token": access_token}

@app.get("/api/doctors")
async def get_doctors(current_user: dict = Depends(get_current_user)):
    try:
        doctors = sorted(
            list(data['doctor_name'].map(lambda x: label_encoders['doctor_name'].inverse_transform([x])[0]).unique())
        )
        return {"doctors": doctors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving doctors: {str(e)}")

@app.get("/api/available-slots")
async def get_available_slots(doctor_name: str, current_user: dict = Depends(get_current_user)):
    if not doctor_name:
        raise HTTPException(status_code=400, detail="Doctor name is required")

    slots = get_next_available_slots(doctor_name)
    return {"slots": slots}

@app.post("/api/appointments")
async def book_appointment(appointment: AppointmentRequest, current_user: dict = Depends(get_current_user)):
    new_appointment = allocate_appointment(
        appointment.doctor_name,
        appointment.patient_name,
        appointment.age,
        appointment.gender,
        appointment.chosen_slot,
        current_user['email']
    )
    return {
        "message": "Appointment successfully scheduled",
        "appointment": new_appointment
    }

@app.get("/api/appointments")
async def get_user_appointments(current_user: dict = Depends(get_current_user)):
    appointments = list(appointmentCol.find({"user_email": current_user['email']}, {'_id': 0}))
    serialized_appointments = jsonable_encoder(appointments)
    return {"appointments": serialized_appointments}