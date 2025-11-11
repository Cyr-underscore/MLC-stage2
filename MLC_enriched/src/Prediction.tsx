import type { PatientState } from "./main"
import Decision from "./Decision";
import WarningDivergingBarChart from "./components/WarningDivergingBarChart";
import "./style/prediction.css"

type Props = {
  selectedPatientId: string;
  updatePatientState: (id: string, newState: Partial<any>) => void;
  currentPatientState: PatientState;
};

export default function Prediction({ selectedPatientId, updatePatientState, currentPatientState }: Props) {
  return (
    <div className="prediction-container">
      <div className="prediction-AI">
        <h1>AI Prediction Explanation</h1>
        {selectedPatientId && (
          <WarningDivergingBarChart selectedPatientId={selectedPatientId} />
        )}
      </div>
      <div style={{ height: '20px' }}>
      </div>
      <Decision
        selectedPatientId={selectedPatientId}
        updatePatientState={updatePatientState}
        currentPatientState={currentPatientState}
      />
    </div>
  );
}
