import "../style/navbar.css";
import { logUserAction } from "../utils/logger";
import { useState } from "react";

interface NavbarProps {
    selectedPatientId?: string | null;
    setSelectedPatientId: (id: string | null) => void;
}

export default function Navbar({ selectedPatientId, setSelectedPatientId }: NavbarProps) {
    const [isAuthenticated, setIsAuthenticated] = useState(() => {
        return localStorage.getItem('isAuthenticated') === 'true';
    });

    const handleLogout = () => {
        const username = localStorage.getItem("username");
        logUserAction("logout", { user: username });
        localStorage.removeItem("token");
        localStorage.removeItem("username");
        localStorage.removeItem("isAuthenticated");
        setIsAuthenticated(false);
        window.location.reload();
    }

    return (
        <nav className="navbar">
            {selectedPatientId ? (
                <>
                    <div className='column-50'>
                        <button className="return-button" onClick={() => {
                            logUserAction("return_to_patient_list", { from_patient_id: selectedPatientId });
                            setSelectedPatientId(null);
                        }}>
                            BACK
                        </button>
                    </div>
                    <h1>Diagnosis: Metastatic Lung Cancer</h1>
                    <div className='column-50'>
                        <button className="logout-button" title='Logout' onClick={handleLogout}>
                            LOG OUT
                        </button>
                    </div>
                </>
            ) : (
                <>
                    <h1>List of Patients</h1>
                    <div className='column-50'>
                        <button className="logout-button" title='Logout' onClick={handleLogout}>
                            LOG OUT
                        </button>
                    </div>
                </>
            )}
        </nav>
    );
}