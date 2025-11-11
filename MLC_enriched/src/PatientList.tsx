import { useEffect, useState } from 'react';
import { fetchPatients } from "./utils/api_getPatients";
import { logUserAction } from "./utils/logger";
import Loader from './components/Loader';
import './style/patientList.css';

interface Patient {
    id_patient: string;
    age: number | string;
    gender: string;
    stroke_type: string;
    prediction_proba?: number | null;
}

interface PatientListProps {
    onSelectPatient: (id: string) => void;
    patientStatuses: Record<string, 'seen' | 'in_progress' | 'not_seen'>;
}

export default function PatientList({ onSelectPatient, patientStatuses }: PatientListProps) {
    const token = localStorage.getItem("token");
    const [patients, setPatients] = useState<Patient[]>([]);
    const [loading, setLoading] = useState(true);
    const [page, setPage] = useState(0);
    const [limit] = useState(15);
    const [total, setTotal] = useState(0);

    const [searchIdInput, setSearchIdInput] = useState('');
    const [ageMinInput, setAgeMinInput] = useState<string>('');
    const [ageMaxInput, setAgeMaxInput] = useState<string>('');

    const [searchIdFilter, setSearchIdFilter] = useState('');
    const [ageMinFilter, setAgeMinFilter] = useState('');
    const [ageMaxFilter, setAgeMaxFilter] = useState('');

    const [isAuthenticated, setIsAuthenticated] = useState(() => {
        return localStorage.getItem('isAuthenticated') === 'true';
    });

    const handleSearch = () => {
        setPage(0);
        setSearchIdFilter(searchIdInput);
        setAgeMinFilter(ageMinInput);
        setAgeMaxFilter(ageMaxInput);
    };

    useEffect(() => {
        setLoading(true);
        fetchPatients({
            page,
            limit,
            searchIdFilter,
            ageMinFilter,
            ageMaxFilter,
            token
        })
            .then(data => {
                if (data.patients) {
                    setPatients(data.patients);
                    setTotal(data.total);
                }
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, [page, searchIdFilter, ageMinFilter, ageMaxFilter]);

    if (loading) return <Loader />;

    const resetFilters = () => {
        setSearchIdInput('');
        setAgeMinInput('');
        setAgeMaxInput('');
        setSearchIdFilter('');
        setAgeMinFilter('');
        setAgeMaxFilter('');
        setPage(0);
    };

    const handlePatientClick = (id_patient: string) => {
        logUserAction("select_patient", { patient_id: id_patient });
        onSelectPatient(id_patient);
    };

    return (
        <div className="home-page">
            <div className="nav-filters">
                <h4>Filters</h4>
                <input
                    type="text"
                    placeholder="Search by ID"
                    value={searchIdInput}
                    onChange={(e) => setSearchIdInput(e.target.value)}
                />
                <input
                    type="number"
                    placeholder="Minimum Age"
                    value={ageMinInput}
                    onChange={(e) => setAgeMinInput(e.target.value)}
                />
                <input
                    type="number"
                    placeholder="Maximum Age"
                    value={ageMaxInput}
                    onChange={(e) => setAgeMaxInput(e.target.value)}
                />
                <div className="research-buttons">
                    <button onClick={resetFilters} className="reset-filters">
                        Reset
                    </button>
                    <button onClick={handleSearch} className="search-button">
                        Search
                    </button>
                </div>
            </div>
            <div className="patient-list-page">
                <table className="patient-list-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Age</th>
                            <th>Status</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {patients.map((patient) => {
                            const status = patientStatuses[patient.id_patient] || 'not_seen';

                            let statusLabel = '';
                            let statusClass = '';

                            switch (status) {
                                case 'seen':
                                    statusLabel = '✅ Seen';
                                    statusClass = 'status-seen';
                                    break;
                                case 'in_progress':
                                    statusLabel = '⏳ In progress';
                                    statusClass = 'status-in-progress';
                                    break;
                                case 'not_seen':
                                    statusLabel = '📥 Not seen';
                                    statusClass = 'status-not-seen';
                                    break;
                            }

                            return (
                                <tr key={patient.id_patient}>
                                    <td>{patient.id_patient}</td>
                                    <td>{typeof patient.age === 'number' ? Math.trunc(patient.age) : patient.age}</td>
                                    <td><span className={statusClass}>{statusLabel}</span></td>
                                    <td>
                                        <button onClick={() => handlePatientClick(patient.id_patient)}>
                                            View Details
                                        </button>
                                    </td>
                                </tr>
                            )
                        })}
                    </tbody>
                </table>
                {patients.length === 0 && (
                    <p style={{ textAlign: 'center', marginTop: '1rem', color: '#555' }}>
                        No patients found. Try adjusting your filters.
                    </p>
                )}
                <div className="pagination-controls">
                    <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>◀</button>
                    <span>{page + 1} / {Math.ceil(total / limit)}</span>
                    <button onClick={() => setPage((p) => (p + 1 < total / limit ? p + 1 : p))} disabled={(page + 1) * limit >= total}>▶</button>
                </div>
            </div>
        </div>
    );
}
