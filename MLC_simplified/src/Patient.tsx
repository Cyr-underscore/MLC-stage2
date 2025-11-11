import SearchBar from "./components/Searchbar"
import { useEffect, useState } from "react";
import { fetchPatientVitals } from "./utils/api_getVitals";
import Loader from "./components/Loader";
import "./style/patient.css"

type PatientVitals = {
    [key: string]: string | number | null;
}

type Props = {
    selectedPatientId: string;
};


export default function Patient({ selectedPatientId }: Props) {
    const [vitals, setVitals] = useState<PatientVitals | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [searchValue, setSearchValue] = useState<string>("");

    useEffect(() => {
        const token = localStorage.getItem("token");
        fetchPatientVitals(selectedPatientId, token)
            .then(({ error, vitals }) => {
                setError(error);
                setVitals(vitals);
            });
    }, [selectedPatientId]);

    const filteredVitals = vitals
        ? Object.entries(vitals).filter(([Key, value]) =>
            Key.toLowerCase().includes(searchValue.toLowerCase()) ||
            (value !== null && value.toString().toLowerCase().includes(searchValue.toLowerCase()))
        )
        : [];

    return (
        <div className="patient-container">
            <div className="patient-overview">
                <div className="patient-header">
                    <h1>Patient {selectedPatientId}: Overview</h1>
                </div>
                <div className="patient-search">
                    <SearchBar searchTitle="Vitals" onSearch={setSearchValue} />
                </div>
                <div className="patient-table">
                    {vitals ? (
                        <table className="patient-info-table">
                            <thead>
                                <tr className="header-table">
                                    <th>Vitals & Informations</th>
                                    <th>Values</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredVitals.map(([key, value], index) => {
                                    const formattedValue =
                                        value !== null
                                            ? typeof value === "number"
                                                ? Number.isInteger(value)
                                                    ? value
                                                    : value.toFixed(2)
                                                : !isNaN(Number(value)) && value !== ""
                                                    ? Number.isInteger(Number(value))
                                                        ? Number(value)
                                                        : Number(value).toFixed(2)
                                                    : value.toString()
                                            : "N/A";

                                    return (
                                        <tr key={index}>
                                            <td title={key}>{key}</td>
                                            <td title={formattedValue.toString()}>{formattedValue}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    ) : (
                        !error && (
                            <div className="loading-message">
                                <Loader />
                            </div>
                        )
                    )}
                    {error && <p className="error-message">{error}</p>}
                </div>
            </div>
        </div>
    )
}