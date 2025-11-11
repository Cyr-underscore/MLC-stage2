import { logUserAction } from "./utils/logger"
import type { PatientState } from "./main"
import { useState } from 'react';
import Modal from "./components/Modal";
import "./style/decision.css"


type Props = {
    selectedPatientId: string;
    updatePatientState: (id: string, newState: Partial<any>) => void;
    currentPatientState: PatientState;
}

export default function Decision({ selectedPatientId, updatePatientState, currentPatientState }: Props) {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedChoice, setSelectedChoice] = useState<string | null>(null);
    const [reason, setReason] = useState("");

    const handleInitialChoice = (selectedChoice: string) => {
        setSelectedChoice(selectedChoice);
        setIsModalOpen(true);
    }

    const handleModalSubmit = () => {
        logUserAction("decision_made", {
            decision: selectedChoice,
            reason: reason
        });

        updatePatientState(selectedPatientId, {
            finalDecision: selectedChoice,
            decisionReason: reason,
            step: 3
        });

        setIsModalOpen(false);
        setReason("");
        setSelectedChoice(null);
    }

    return (
        <div className="decision-container">
            <h3>Do you agree with the AI's reasoning and prediction for this patient?</h3>
            <div className="buttons">
                <button className="disagree-button" onClick={() => handleInitialChoice("Disagree")}>I Disagree</button>
                <button className="dk-button" onClick={() => handleInitialChoice("Not Sure")}>I'm Not Sure</button>
                <button className="agree-button" onClick={() => handleInitialChoice("Agree")}>I Agree</button>
            </div>

            <Modal isOpen={isModalOpen}>
                <h4>Please briefly explain your choice:</h4>
                <textarea
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                    placeholder={`What made you ${selectedChoice} with the AI prediction?"`}
                    rows={4}
                />
                <div className="modal-buttons">
                    <button onClick={handleModalSubmit} className="submit-button">Submit</button>
                </div>
            </Modal>
        </div>
    )
}