"""
======================================================
    FastAPI Medical Backend API
    - Authenticated endpoints for patient data
    - SHAP explanations and AI-generated summaries
    - Includes TinyLlama (phi-3-medical) LLM fallback
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
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")


# -------------------- FastAPI Setup ------------------
app = FastAPI(title="Medical AI API", version="1.0")

# Security
security = HTTPBearer()
API_TOKEN = "M58-L35-C62"

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
    print(f"🌐 Access your API at http://{local_ip}:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


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


### --------------- Medical LLM Summary Generator ---------------- ###


# -------------------- MEDICAL LLM --------------------
class MedicalSummaryGenerator:
    """Handles AI-based medical summaries using TinyLlama or fallback model"""

    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.load_model()

    def _setup_device(self) -> str:
        """Detect GPU availability"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🎯 GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM)")
                return "cuda" if vram_gb >= 4 else "cpu"
            except Exception as e:
                print(f"❌ GPU error: {e}")
                return "cpu"
        print("❌ CUDA not available - CPU mode activated")
        return "cpu"

    def load_model(self):
        """Load main or fallback model"""
        print(f"🔮 Loading {self.model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            if self.device == "cpu":
                self.model.to("cpu")
                torch.set_num_threads(min(4, torch.get_num_threads()))

            print(f"✅ Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"❌ Loading error: {e}")
            self._load_ultralight_model()

    def _load_ultralight_model(self):
        """Fallback lightweight model"""
        self.model_name = "microsoft/DialoGPT-small"
        print(f"🔄 Fallback to {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to("cpu")
        print("✅ Fallback model loaded")
            
    def generate_medical_summary(self, patient_data: Dict) -> str:
        import time
        import threading
        
        def generation_with_timeout():
            nonlocal result, generation_error, completed
            
            try:
                # English prompt
                prompt = self.build_english_prompt(patient_data)
                
                print("=" * 80)
                print("📝 PROMPT ENGLISH:")
                print(prompt)
                print("=" * 80)
                
                # Tokenization for TinyLlama
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=True
                )
                
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=400,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2
                    )
                
                # Decode the generated text
                full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("📄 FULL GENERATED TEXT:")
                print(full_generated_text)
                
                # Extract only the part after the prompt
                raw_result = full_generated_text[len(prompt):].strip()
                
                # Clean up any incomplete sentences or cut-off text
                if raw_result:
                    last_period = raw_result.rfind('.')
                    if last_period != -1:
                        raw_result = raw_result[:last_period + 1]
                    
                    if len(raw_result.split()) < 3:
                        raw_result = "Unable to generate comprehensive summary."
                
                # Format the result as HTML with CSS classes
                result = self._format_summary_html(raw_result)
                
                print(f"🎯 EXTRACTED RESULT: '{raw_result}'")
                print(f"🎯 HTML FORMATTED RESULT: '{result}'")
                
                completed = True
                
            except Exception as e:
                generation_error = str(e)
                print(f"💥 ERROR: {generation_error}")
                completed = True
        
        result = ""
        generation_error = None
        completed = False
        
        print("🚀 STARTING MEDICAL SUMMARY GENERATION")
        start_time = time.time()
        
        thread = threading.Thread(target=generation_with_timeout)
        thread.daemon = True
        thread.start()
        
        thread.join(timeout=45.0)
        
        generation_time = time.time() - start_time
        print(f"⏱️  Total time: {generation_time:.1f}s")
        
        if not completed:
            return self._format_error_html("Timeout - Model took too long to respond")
        
        if generation_error:
            return self._format_error_html(f"Error: {generation_error}")
        
        return result if result else self._generate_fallback_summary_html(patient_data)

    def _format_summary_html(self, summary_text: str) -> str:
        """Format the AI summary as plain text"""
        if not summary_text or summary_text == "Unable to generate comprehensive summary.":
            return self._format_error_text("No summary generated by the AI model")
        
        # Retourner le texte brut directement
        return summary_text.strip()

    def _format_error_html(self, error_message: str) -> str:
        """Format error messages as plain text"""
        return f"Error: {error_message}"

    def _generate_fallback_summary_html(self, patient_data: Dict) -> str:
        """Generate fallback summary as plain text"""
        print("🔄 Generating fallback with complete data...")
        
        # Extract most important data for fallback (same logic as before)
        important_data = {}
        
        # Demographics
        age = next((v for k, v in patient_data.items() if 'age' in k.lower() and v is not None), None)
        gender = next((v for k, v in patient_data.items() if any(word in k.lower() for word in ['sex', 'gender']) and v is not None), None)
        
        if age:
            important_data['age'] = age
        if gender:
            important_data['gender'] = 'Male' if gender == 1 else 'Female'
        
        # Cancer
        cancers = [k for k, v in patient_data.items() if any(word in k.lower() for word in ['diag_', 'met_']) and v == 1]
        if cancers:
            important_data['cancers'] = cancers[:3]
        
        # Present symptoms
        symptoms_present = [k for k, v in patient_data.items() if any(word in k.lower() for word in ['pain', 'nausea', 'fatigue']) and v == 1]
        if symptoms_present:
            important_data['symptoms'] = symptoms_present
        
        # Important vitals
        vital_keys = [k for k in patient_data.keys() if k.startswith('vitals_') and not k.endswith('_diff')]
        important_vitals = {}
        for key in vital_keys[:5]:
            if patient_data.get(key) is not None:
                important_vitals[key] = patient_data[key]
        
        if important_vitals:
            important_data['key_vitals'] = important_vitals
        
        # Build fallback summary text
        parts = ["Medical summary (basic analysis):"]
        
        if important_data.get('age') and important_data.get('gender'):
            parts.append(f"Patient: {important_data['age']} years old, {important_data['gender']}")
        
        if important_data.get('cancers'):
            parts.append(f"Diagnosis: {', '.join(important_data['cancers'])}")
        
        if important_data.get('symptoms'):
            parts.append(f"Symptoms: {', '.join(important_data['symptoms'])}")
        
        if important_data.get('key_vitals'):
            vital_str = ", ".join([f"{k}: {v}" for k, v in important_data['key_vitals'].items()])
            parts.append(f"Vitals: {vital_str}")
        
        fallback_text = "\n".join(parts) + "\n\n[Note: Basic analysis - AI model not available]"
        print(f"📄 Fallback generated: {fallback_text}")
        
        return fallback_text

    def build_english_prompt(self, patient_data: Dict) -> str:
        # Mapping des variables vers des noms explicites
        variable_mapping = {
            "age": "Age",
            "male": "Gender (0=F, 1=M)",
            "days_from_diag": "Days since diagnosis",
            "diag_DC18": "Colorectal cancer",
            "diag_DC34": "Lung cancer",  
            "diag_DC50": "Breast cancer",
            "diag_DC79": "Metastases",
            "comorb_met": "Metastatic disease",
            "comorb_CKD": "Chronic kidney disease",
            "comorb_COPD": "COPD",
            "comorb_diabetes1": "Diabetes",
            "blood_test_NPU01349": "Hemoglobin",
            "blood_test_NPU01685": "Creatinine",  
            "blood_test_NPU03011": "CRP",
            "vitals_bmi": "BMI",
            "side_effect_fatigue": "Fatigue",
            "side_effect_pain": "Pain", 
            "side_effect_anemia": "Anemia",
            "drug_prn0_L01AA01": "Chemotherapy",
            "RT_BWGC1": "Radiotherapy",
            "surgery_KGEA00": "Surgery"
        }
        
        # Format des données avec noms explicites
        data_items = []
        for key, display_name in variable_mapping.items():
            value = patient_data.get(key)
            if value is not None and value != '':
                # Format simplifié pour économiser des tokens
                data_items.append(f"{display_name}: {value}")
        
        formatted_data = " | ".join(data_items)
        
        prompt = f"""Context: You are a medical AI assistant in oncology. Analyze patient data and provide a concise clinical summary.

Guidelines:
- 0=No/Absent, 1=Yes/Present
- BMI <18.5=underweight, CRP>1=inflammation
- Focus on cancer status, key findings, and recommendations
- Fo use to specify your name or role as a assistant

Patient data: {formatted_data}

Medical summary: The patient is a"""
        
        return prompt

summary_generator = MedicalSummaryGenerator()




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

        # Cette fonction retourne maintenant du texte brut
        text_summary = summary_generator.generate_medical_summary(patient_data)
        
        return {"summary": text_summary}  # ← Texte brut dans la réponse JSON

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate medical summary: {str(e)}"}