
# MLC_Stage2

This repo is base on an existing project from @anais-mez an the repo : https://github.com/anais-mez/MLC

MLC provides an interactive visual interface to help clinicians understand **SHAP  (SHapley Additive exPlanations)** values and their influence on machine learning predictions.
It  was developed as part of a research study to evaluate **wheter presentign SHAP explanations in a tailored interface could improve clinicians' understanding of model decisions and reduce overtrust** in AI systems.

## 📌 Table of Contents

- [📚 Project Overview](#-project-overview)
- [🧪 Project Purpose](#-project-purpose)
- [🏗 Project Structure](#-project-structure)
- [⚙️ Installation Instructions](#️-installation-instructions)
    - [🔧 1. Clone the repository](#-1-clone-the-repository)
    - [🧪 2. Backend Setup (FastAPI + SHAP)](#-2-backend-setup-fastapi--shap)
    - [🖥️ 3. Frontend Setup (React + Vite)](#️-3-frontend-setup-react--vite)
    - [🔐 Authentication](#-authentication)
- [🏗 Backend (FastAPI + SHAP)](#-backend-fastapi--shap)
- [🖥 Frontend (React + TypeScript)](#-frontend-react--typescript)
- [❗ Disclaimer](#-disclaimer)


## 📚 Project Overview

- 🔬 **Goal**: Support interpretability and critical thinking when using AI in clinical decision-making
- 📊 **Frontend**: Interactive SHAP charts with custom guidance
- 🧠 **Backend**: API delivering model predictions and SHAP explanations
- 🔐 **Authentication**: JWT-based user login and logging

---

## 🧪 Project Purpose

The core research questions are:

- **Does visualizing SHAP values improve interpretability** for medical professionals?
- **Can proper explanations reduce the risk of overtrust** in AI predictions?

To explore these questions, this tool presents SHAP values through interactive charts, textual guidance, and contextual warnings to support critical thinking and reflection.

---

##  🏗 Project Structure

```
MLC/
├── API/ # 🧠 FastAPI backend
│ ├── main.py # Main API routes
│ ├── scripts/ # ML model, SHAP logic (e.g. model.joblib)
│ ├── data/ # Dataset, mapping files, user logs
│ └── requirements.txt # Python dependencies
│
├── MLC_enriched/ # 🟢 React app with SHAP explanations (Vite)
│ ├── src/ # Source files (components, views, logic)
│ ├── public/ # Static assets
│ └── package.json # JS dependencies
│
├── MLC_simplified/ # 🟠 React app with basic SHAP only (Vite)
│ ├── src/ # Source files
│ ├── public/ # Static assets
│ └── package.json # JS dependencies
│
└── README.md # You're here 🙂
```

---

## ⚙️ Installation Instructions

> ⚠️ **Python 3.11 is required**, **3.11.9 is recommended** for the backend to work with current dependencies. Other version may cause incompatibilities with packages like SHAP and the model.

### 🔧 1. Clone the repository

```bash
git clone https://github.com/anais-mez/MLC.git
cd MLC
```

### 🧪 2. Backend Setup (FastAPI + SHAP)

```bash
cd API
pip install -r requirements2.txt

# In case you have a 1060 6Go this torch work :
# other wise install the lastest driver of your gpu and found a compatible version of cuda and download the coresponding pytorch package.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Run the API
uvicorn main:app --reload
```

By default, the API runs at:
http://127.0.0.1:8000

✅ Make sure your ``scripts/model.joblib`` file exists, or retrain and save the model.

### 🖥️ 3. Frontend Setup (React + Vite)

```bash
cd ../MLC_enriched #or ../MLC_simplified
npm install
npm run dev
```

This starts the frontend at:
http://127.0.0.1:4000

For the MLC_simplified, do the same things and this starts the frontend at:
http://127.0.0.1:5173

### 🔐 Authentication

This application uses **JWT (JSON Web Token)** for authenticating API requests via the `Authorization` header using the **Bearer** scheme.

To function correctly, the frontend expects the following values to be present in the browser's `localStorage`:

- `token`: the JWT used to authenticate API requests
- `username`: the username of the logged-in user

These values are used to:

- ✅ Add the header `Authorization: Bearer <token>` to all API requests
- 📝 Log user interactions (e.g., tab switches) via authenticated calls to the logging endpoint

---

## 🏗 Backend (FastAPI + SHAP)

### Structure of the API

```
API/
├── main.py               # FastAPI app and route definitions
├── scripts/              # ML model loading, prediction, and SHAP logic
│   └── model.joblib      # Pre-trained machine learning model
├── data/                 # Input datasets, mappings, and logs
│   ├── sample_20_false.csv   # Synthetic patient data with incorrect SHAP values (test case)
│   ├── sample_20.csv         # Synthetic patient data with correct SHAP values
│   ├── mapping.csv           # Feature name/value mapping for SHAP visualization
│   └── logs_actions.json     # User interaction logs (e.g., clicks, tab switches)
└── requirements.txt      # Python dependencies for running the API
```
> The backend is responsible for providing predictions form the trained machine learning model and computing SHAP values to interpret those predictions.

### FastAPI API Overview

- **Main endpoint**: ``/predict`` - To get model predictions based on the input data.
- **SHAP explanations**: ``/shap`` - To get SHAP values for model predictions.
- **User Logging**: Each user's interactions are logged with their token for activity tracking.

Example FastAPI endpoint:
```python
@app.get("/vitals/{id_patient}")
def get_values(id_patient: str, auth: HTTPAuthorizationCredentials = Depends(authenticate)):
    try:
        patient_rows = df_all_patients[df_all_patients['id_patient'] == id_patient]
        if patient_rows.empty:
            return {"error": "Patient not found"}

        vitals = patient_rows.iloc[0].to_dict()
        vitals = {k: (None if pd.isna(v) else v) for k, v in vitals.items()}
        
        vitals_mapped = {
            column_mapping.get(k, k): v 
            for k, v in vitals.items() if k in column_mapping
        }

        return {
            "id_patient": id_patient,
            "vitals": vitals_mapped
        }
        
    except Exception as e:
        return {"error": str(e)}
```

---

---

## 🖥 Frontend (React + TypeScript)

### React App Structure

The frontend is built using **React** with **TypeScript**. The main parts of the app are:
1. ``src/components/`` – Reusable UI components such as SHAP charts, modals, and layout elements.
2. ``src/utils/`` – Utility functions, including API handlers, logging tools, and helper methods.
3. ``src/styles/`` – Global and component-specific CSS styles.
4. ``src/assets/`` – Static assets such as images, icons, and logos.

Example API call in React (using ``fetch``):
```TypeScript
useEffect(() => {
        const token = localStorage.getItem("token");
        fetch(`http://127.0.0.1:8000/vitals/${selectedPatientId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
            .then((res) => res.json())
            .then((data) => {
                if (data.error) {
                    setError(data.error);
                    setVitals(null);
                } else {
                    setVitals(data.vitals);
                    setError(null);
                }
            })
            .catch((err) => {
                setError("Erreur lors de la récupération des données.");
                console.error(err);
            });
    }, [selectedPatientId]);
```

The frontend interacts with the FastAPI backend to retrieve model predictions and SHAP values, which are then displayede as interactive visualizations.

## ❗ Disclaimer

This tool is for **research and educational purposes only**. It does **not provide clinical advice**, and its predictions or explanations should not be used to make real-world decisions.

Always apply professional and clinical judgment.