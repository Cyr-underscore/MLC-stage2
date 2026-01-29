"""
======================================================
    FastAPI Medical Backend API
    - Authenticated endpoints for patient data
    - SHAP explanations and AI-generated summaries
    - Includes TinyLlama LLM fallback
======================================================
"""

# -------------------- Imports ------------------------
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from functools import lru_cache
from typing import Optional, Dict
import torch
import shap
import joblib
import json
import pandas as pd
import os
import socket
import warnings

from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from AI_stuff import MedicalSummaryGenerator 

import datetime
from MouseTracking import MouseData


warnings.filterwarnings("ignore")


# -------------------- FastAPI Setup ------------------
app = FastAPI(title="Medical AI API", version="1.0")

# Security
security = HTTPBearer()
API_TOKEN = "M58-L35-C62"   # have to change the way the tsx and the api connect this before production 

def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token-based authentication"""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )

# -------------------- CORS Config --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Root endpoint: API health check"""
    return {"message": "Welcome to the Medical AI API"}

# -------------------- Run Local ----------------------
if __name__ == "__main__":
    import uvicorn
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"🌐 Access your API at http://{local_ip}:8000")      # change the adress to a real adress variable of the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)     # make it callable from outside the server


# -------------------- LOGGING ------------------------
ACTION_FILE = "data/logs_actions.json"

def load_actions() -> dict:
    """Load user action logs"""
    if not os.path.exists(ACTION_FILE):
        return {}
    try:
        with open(ACTION_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_actions(data: dict):
    """Save user action logs"""
    with open(ACTION_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.post("/log_action")
def log_action(payload: dict, auth: HTTPAuthorizationCredentials = Depends(authenticate)):
    """Log user actions with username, action, and timestamp"""
    try:
        data = load_actions()
        username = payload.get("username")
        action = payload.get("action")
        timestamp = payload.get("timestamp")
        details = payload.get("details", {})

        if not username or not action or not timestamp:
            return {"error": "Missing username, action or timestamp"}

        data.setdefault(username, []).append({
            "action": action,
            "timestamp": timestamp,
            "details": details
        })

        save_actions(data)
        return {"status": "ok"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# -------------------- DATASET & MODEL ----------------
df_full = pd.read_csv("data/sample_20.csv")
df_full["id_patient"] = df_full["id_patient"].astype(str)

df_fake = pd.read_csv("data/sample_20_false.csv")
df_fake["id_patient"] = df_fake["id_patient"].astype(str)

# Combine real & fake datasets
df_all_patients = pd.concat([df_full, df_fake], ignore_index=True).drop_duplicates(subset=["id_patient"])

# Load prediction model
model = joblib.load("scripts/model.joblib")

# Columns to drop before prediction/imputation
cols_to_drop = [
    "id_patient", "days_from_diag", "contact", "bl_comorb_CVA", "bl_comorb_connect",
    "bl_comorb_diabetes1", "bl_comorb_infarct", "bl_comorb_met", "bl_comorb_ulcer",
    "bl_comorb_vascular", "bl_diag_DC18", "bl_diag_DC44", "bl_diag_DC50",
    "bl_diag_DC61", "bl_diag_DC67", "bl_side_effect_anemia", "bl_side_effect_bact_inf",
    "bl_side_effect_constip", "bl_side_effect_diarrhea", "drug_prn0_A02BC05_cumul",
    "drug_prn0_A03FA01_cumul", "drug_prn0_L01CB01_cumul", "drug_prn0_R06AA04_cumul",
    "RT_BWGC1_30d", "drug_prn0_A02BA02_30d"
]

# Train imputer on complete dataset
X_train_for_imputer = df_full.drop(columns=cols_to_drop, errors="ignore").apply(pd.to_numeric, errors="coerce")
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train_for_imputer)

def load_column_mapping(filepath: str) -> dict:
    """Load human-readable names for variables"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

column_mapping = load_column_mapping("data/mapping.csv")



# -------------------- API ENDPOINTS ------------------
@lru_cache(maxsize=1)
def get_patient_table():
    """Return cached list of patients (id, age)"""
    display_columns = ["id_patient", "age"]
    for col in display_columns:
        if col not in df_all_patients.columns:
            df_all_patients[col] = None
    return df_all_patients[display_columns].fillna("").to_dict(orient="records")



@app.get("/shap/{id_patient}")
def compute_shap(id_patient: str, auth: HTTPAuthorizationCredentials = Depends(authenticate)):
    try:
        id_patient = str(id_patient).strip()
        
        if id_patient in df_fake['id_patient'].values:
            row = df_fake[df_fake['id_patient'] == id_patient]
            if 'shap_data' in row.columns and pd.notna(row.iloc[0]['shap_data']):
                shap_data = json.loads(row.iloc[0]['shap_data'])
                return shap_data
            else:
                return {"error": "SHAP values not found"}
            
        elif id_patient in df_full['id_patient'].values:
            row = df_full[df_full['id_patient'] == id_patient]
            
            X_patient = row.drop(columns=cols_to_drop, errors='ignore')
            X_patient = X_patient.apply(pd.to_numeric, errors='coerce')

            # SHAP
            explainer = shap.Explainer(model)
            shap_values = explainer(X_patient)
            shap_values_patient = shap_values[0]            

            shap_df = pd.DataFrame({
                'feature': X_patient.columns,
                'shap_value': shap_values_patient.values,
                'feature_value': X_patient.iloc[0].values
            })
            shap_df['abs_val'] = shap_df['shap_value'].abs()
            shap_df_sorted = shap_df.sort_values(by='abs_val', ascending=False).head(10)
            shap_df_sorted['label'] = shap_df_sorted.apply(
                lambda row: f"{row['feature']}: {row['feature_value']}", axis=1
            )

            shap_data = []
            for _, row in shap_df_sorted.iterrows():
                feature_name = column_mapping.get(row['feature'], row['feature'])
                shap_data.append({
                    'feature': feature_name,
                    'shap_value': float(row['shap_value']),
                    'feature_value': float(row['feature_value']),
                    'label_with_value': row['label']
                })

            return {
                "id_patient": id_patient,
                "shap_values": shap_data
            }
            
        else:
            return {"error": "Patient not found in either dataset."}
    except Exception as e:
        return {"error": str("shap error: " + str(e))}
    

@app.get("/patients")
def list_patients(
    skip: int = 0,
    limit: int = 50,
    id_patient: Optional[str] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    auth: HTTPAuthorizationCredentials = Depends(authenticate),
):
    """List patients with optional filters"""
    try:
        df_filtered = df_all_patients[["id_patient", "age"]].copy()

        if id_patient:
            df_filtered = df_filtered[df_filtered["id_patient"].str.contains(id_patient, case=False, na=False)]
        if age_min is not None:
            df_filtered = df_filtered[df_filtered["age"].apply(lambda x: isinstance(x, (int, float)) and x >= age_min)]
        if age_max is not None:
            df_filtered = df_filtered[df_filtered["age"].apply(lambda x: isinstance(x, (int, float)) and x <= age_max)]

        df_filtered["id_patient"] = df_filtered["id_patient"].astype(str)
        df_filtered = df_filtered.sort_values(by="id_patient")

        total = len(df_filtered)
        patients = df_filtered.iloc[skip: skip + limit].fillna("").to_dict(orient="records")

        return {"patients": patients, "total": total}
    except Exception as e:
        return {"error": str(e)}

@app.get("/vitals/{id_patient}")
def get_values(id_patient: str, auth: HTTPAuthorizationCredentials = Depends(authenticate)):
    """Return patient vitals mapped with readable names"""
    try:
        patient = df_all_patients[df_all_patients["id_patient"] == id_patient]
        if patient.empty:
            return {"error": "Patient not found"}

        vitals = patient.iloc[0].to_dict()
        vitals = {k: (None if pd.isna(v) else v) for k, v in vitals.items()}
        vitals_mapped = {column_mapping.get(k, k): v for k, v in vitals.items() if k in column_mapping}

        return {"id_patient": id_patient, "vitals": vitals_mapped}
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict/{id_patient}")
def predict_patient(id_patient: str, auth: HTTPAuthorizationCredentials = Depends(authenticate)):
    """Compute model prediction for a given patient"""
    try:
        id_patient = str(id_patient).strip()

        # Use precomputed fake patients
        if id_patient in df_fake["id_patient"].values:
            row = df_fake[df_fake["id_patient"] == id_patient]
            return {
                "id_patient": id_patient,
                "prediction_proba": float(row.iloc[0]["prediction_proba"]),
                "prediction_class": int(row.iloc[0]["prediction_class"]),
            }

        # Real patients
        elif id_patient in df_full["id_patient"].values:
            row = df_full[df_full["id_patient"] == id_patient]
            X_patient = row.drop(columns=cols_to_drop, errors="ignore").apply(pd.to_numeric, errors="coerce")
            X_patient = X_patient.reindex(columns=X_train_for_imputer.columns)
            X_patient_imputed = pd.DataFrame(imputer.transform(X_patient), columns=X_patient.columns)

            prediction_proba = model.predict_proba(X_patient_imputed)[0][1]
            prediction_class = model.predict(X_patient_imputed)[0]

            return {
                "id_patient": id_patient,
                "prediction_proba": round(float(prediction_proba), 3),
                "prediction_class": int(prediction_class),
            }

        else:
            return {"error": "Patient not found"}
    except Exception as e:
        return {"error": str(e)}
    
##--------------- intance of the Summary generator -------------------##
summary_generator = MedicalSummaryGenerator(Big_LLM=True)
##--------------------------------------------------------------------##


##----------------- endpoint for the AI summary -----------------------##

@app.get("/medical-summary/{id_patient}")
def get_medical_summary(id_patient: str, auth: HTTPAuthorizationCredentials = Depends(authenticate)):
    """Generate AI or fallback medical summary for a patient"""
    try:
        print(f"🎯 Request for patient {id_patient}")
        patient = df_all_patients[df_all_patients["id_patient"] == id_patient]
        if patient.empty:
            return {"error": "Patient not found"}

        patient_data = patient.iloc[0].to_dict()
        patient_data = {k: (None if pd.isna(v) else v) for k, v in patient_data.items()}

        # Utilisation de l’instance globale
        text_summary = summary_generator.generate_medical_summary(patient_data)

        return {"summary": text_summary}  # ← JSON response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate medical summary: {str(e)}"}
    
@app.post("/api/mouse-tracking")
async def track_mouse(
    mouse_data: MouseData,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Endpoint pour recevoir les données de tracking de souris
    """
    try:
        # Charger les logs existants
        actions_data = load_actions()
        
        # Créer un ID unique pour cette session
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sauvegarder les données
        actions_data[session_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user": "authenticated_user",  # Vous pouvez récupérer l'utilisateur du token
            "data": mouse_data.dict(),
            "page": mouse_data.url,
            "patient_id": mouse_data.url.split("/")[-1] if "patient" in mouse_data.url else None
        }
        
        # Sauvegarder dans le fichier
        save_actions(actions_data)
        
        print(f"📊 Mouse tracking data received: {mouse_data.dict()}")
        
        return {
            "status": "success",
            "message": "Mouse data saved successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        print(f"❌ Error saving mouse data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving mouse data: {str(e)}"
        )