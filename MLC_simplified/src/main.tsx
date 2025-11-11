import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { useEffect } from 'react'
import { logUserAction } from "./utils/logger";
import './style/index.css'
import Login from './Login'
import PatientList from './PatientList'
import Patient from './Patient'
import Prediction from './Prediction'
import Slider from './components/Slider'
import Navbar from './components/Navbar';
import Modal from './components/Modal';

export type PatientState = {
  step: number;
  decisionMade: boolean;
  decisionError: boolean;
  initialDecision: number | null;
  showAIPrediction: boolean;
  probability: number | null;
  finalDecision?: string;
}


function DashboardApp() {
  type PatientStateMap = Record<string, PatientState>;

  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return localStorage.getItem('isAuthenticated') === 'true';
  });
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [patientsState, setPatientsState] = useState<PatientStateMap>({});
  const patientStatuses: Record<string, 'seen' | 'in_progress' | 'not_seen'> = {};
  const [isModalOpen, setIsModalOpen] = useState(false);

  const getOrInitPatientState = (id: string): PatientState => {
    return patientsState[id] || {
      step: 0,
      decisionMade: false,
      decisionError: false,
      initialDecision: null,
      showAIPrediction: false,
      probability: null,
    }
  }

  const updatePatientState = (id: string, newState: Partial<PatientState>) => {
    setPatientsState((prev) => ({
      ...prev,
      [id]: {
        ...getOrInitPatientState(id),
        ...newState,
      },
    }));
  };

  const currentPatientState = selectedPatientId ? getOrInitPatientState(selectedPatientId) : null;

  useEffect(() => {
    if (!selectedPatientId) return;

    const token = localStorage.getItem("token");

    fetch(`http://localhost:8000/predict/${selectedPatientId}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
      .then(res => res.json())
      .then(data => {
        console.log("Prediction data received:", data);
        if (data.prediction_proba !== undefined) {
          updatePatientState(selectedPatientId!, { probability: data.prediction_proba });
        }
      })
      .catch(err => {
        console.error("Error fetching prediction:", err);
        updatePatientState(selectedPatientId!, { probability: null });
      });
  }, [selectedPatientId]);

  useEffect(() => {
    if (currentPatientState?.step === 1) {
      updatePatientState(selectedPatientId!, { decisionMade: false });
    }
  }, [currentPatientState?.step]);

  if (!isAuthenticated) {
    return <Login onLoginSuccess={() => setIsAuthenticated(true)} />;
  }

  Object.entries(patientsState).forEach(([id, state]) => {
    if (state.finalDecision !== undefined) {
      patientStatuses[id] = 'seen';
    } else if (state.step > 0) {
      patientStatuses[id] = 'in_progress';
    } else {
      patientStatuses[id] = 'not_seen';
    }
  });

  if (!selectedPatientId) {
    return (
      <>
        <Navbar
          selectedPatientId={selectedPatientId}
          setSelectedPatientId={setSelectedPatientId}
        />
        <PatientList
          onSelectPatient={(id: string) => setSelectedPatientId(id)}
          patientStatuses={patientStatuses}
        />
      </>
    );
  }

  const handleNextStepWithValidation = () => {
    if (!currentPatientState?.decisionMade) {
      updatePatientState(selectedPatientId!, { decisionError: true });
    } else {
      setIsModalOpen(true);
    }
  };

  const handleConfirmDecision = () => {
    logUserAction("initial_decision", {
      id_patient: selectedPatientId,
      decision: currentPatientState?.initialDecision,
    });

    updatePatientState(selectedPatientId!, {
      decisionError: false,
      step: (currentPatientState?.step ?? 0) + 1,
    });

    setIsModalOpen(false);
  };

  const handleCancel = () => {
    setIsModalOpen(false);
  }

  return (
    <StrictMode>
      <Navbar
        selectedPatientId={selectedPatientId}
        setSelectedPatientId={setSelectedPatientId}
      />
      <div className="dashboard">
        <div className="data-view">
          {currentPatientState?.step === 3 ? (
            <div className="final-decision-container">
              <div className='final-decision'>
                <h2>Final Decision</h2>
                <p>Your final decision is: <strong>{currentPatientState?.finalDecision}</strong></p>
                <p className="info-message">
                  ✅ You have already reviewed this patient.<br />
                  👉 Please go back and select a new patient to continue.
                </p>
              </div>
            </div>
          ) : (
            <>
              <Patient selectedPatientId={selectedPatientId} />

              {currentPatientState && currentPatientState.step >= 1 &&
                <Prediction
                  selectedPatientId={selectedPatientId}
                  updatePatientState={updatePatientState}
                  currentPatientState={currentPatientState}
                />
              }

              {currentPatientState?.step === 0 && (
                <div className="step0">
                  <p>
                    Check the patient information below.
                    Take the time to fully understand the data.
                    You will need to predict the patient's risk of death within the next 30 days. <br />  <br />
                    Record your initial decision based on the patient data.
                    You will later compare it with the model’s prediction. <br />
                    Please estimate the <strong>risk of death within the next 30 days</strong> for this patient.
                  </p>
                  <Slider onDecisionMade={(value) => {
                    updatePatientState(selectedPatientId!, { initialDecision: value, decisionMade: true, decisionError: false });
                  }} />
                  <button className="button-step" onClick={handleNextStepWithValidation}>
                    Next step, you will see the AI prediction 🢂
                  </button>
                  {currentPatientState?.decisionError && (
                    <p className="error-message">
                      ⚠️ Please make a decision before proceeding.
                    </p>
                  )}
                </div>
              )}

              {currentPatientState?.step === 1 && (
                <div className="step2">
                </div>
              )}
            </>
          )}
        </div>
      </div>
      <Modal isOpen={isModalOpen} onClose={handleCancel}>
        <p>Are you sure you want to submit your initial decision? You won't be able to change it later.</p>
        <p>Your initial decision: {currentPatientState?.initialDecision ? currentPatientState.initialDecision + "%" : 'I don\'t know'}</p>
        <div className="modal-buttons">
          <button onClick={handleConfirmDecision}>Confirm</button>
        </div>
      </Modal>
    </StrictMode>
  );
}

createRoot(document.getElementById('root')!).render(<DashboardApp />);
