import SearchBar from "./components/Searchbar"
import { useEffect, useState } from "react";
import { fetchPatientVitals } from "./utils/api_getVitals";
import { fetchMedicalSummary } from "./utils/api_getMedicalSummary"; 
import Loader from "./components/Loader";
//import "./style/patient.css";
// Remplacer l'import unique par les imports séparés

import "./style/patient/patient.css";
import "./style/patient/patient-table.css";
import "./style/patient/patient-tabs.css";
import "./style/patient/patient-search.css";
import "./style/patient/colors.css";

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
    const [showZeroValues, setShowZeroValues] = useState<boolean>(false);
    const [activeTab, setActiveTab] = useState<"table" | "summary" | "ai_summary">("table");
    const [aiSummary, setAiSummary] = useState<string | null>(null);
    const [aiLoading, setAiLoading] = useState<boolean>(false); 
    const [aiError, setAiError] = useState<string | null>(null);
    
    useEffect(() => {
        const token = localStorage.getItem("token");
        fetchPatientVitals(selectedPatientId, token)
            .then(({ error, vitals }) => {
                setError(error);
                setVitals(vitals);
            });
    }, [selectedPatientId]);

    useEffect(() => {
        if (activeTab === "ai_summary" && !aiSummary && !aiLoading) {
            generateAISummary();
        }
    }, [activeTab]);

        const generateAISummary = async () => {
            if (!selectedPatientId) {
                setAiError("Please select a patient first");
                return;
            }

            setAiLoading(true);
            setAiError(null);
            setAiSummary(null);
            const token = localStorage.getItem("token");
            
            console.log("🔍 [STATE DEBUG] Before API call - aiError:", aiError, "aiSummary:", aiSummary);
            
            try {
                const { error, summary } = await fetchMedicalSummary(selectedPatientId, token);
                
                console.log("🔍 [STATE DEBUG] After API call - error:", error, "summary:", summary);
                
                if (error) {
                    console.log("🔍 [STATE DEBUG] Setting error:", error);
                    setAiError(error);
                } else {
                    console.log("🔍 [STATE DEBUG] Setting summary");
                    setAiSummary(summary);
                }
            } catch (err) {
                console.error("❌ Unexpected error:", err);
                setAiError("Failed to generate summary");
            } finally {
                setAiLoading(false);
            }
        };

    const filteredVitals = vitals
        ? Object.entries(vitals).filter(([key, value]) => {
            const matchesSearch = 
                key.toLowerCase().includes(searchValue.toLowerCase()) ||
                (value !== null && value.toString().toLowerCase().includes(searchValue.toLowerCase()));
            
            const isZeroValue = value === 0 || value === "0" || value === 0.0;
            
            if (!showZeroValues && isZeroValue) {
                return false;
            }
            
            return matchesSearch;
        })
        : [];

    // Fonction pour générer le résumé des données - UNE SEULE FOIS
    const generateSummary = () => {
        if (!vitals || filteredVitals.length === 0) {
            return "No data available for summary.";
        }

        // Extraire les données spécifiques demandées
        const age = vitals.Age || vitals['Age (baseline)'];
        const gender = vitals['Male sex'] === 1 ? 'Male' : 'Female';
        const bmi = vitals['BMI i kg.m<sup>-2</sup>'];
        const metastasis = vitals.Metastasis || vitals['Metastasis (baseline)'];
        const copd = vitals.COPD || vitals['COPD (baseline)'];
        const albumin = vitals['Albumin i g/L'];
        const albuminDiff = vitals['Albumin i g/L (difference)'];
        const alkalinePhosphatase = vitals['Alkaline phosphatase i U/L'] || vitals['Alkaline phosphatase i U/l'];
        const alkalinePhosphataseDiff = vitals['Alkaline phosphatase i U/L (difference)'] || vitals['Alkaline phosphatase i U/l (difference)'];
        const oxygen = vitals.Oxygen || vitals['Oxygen i kPa'];
        const oxygenDiff = vitals['Oxygen (difference)'] || vitals['Oxygen i kPa (difference)'];
        const haemoglobin = vitals['Haemoglobin i mmol/L'];
        const haemoglobinDiff = vitals['Haemoglobin i mmol/L (difference)'];
        const carbonDioxide = vitals['Carbon dioxide i mmol/L'];
        const pain = vitals.Pain || vitals['Pain (baseline)'];
        const nausea = vitals.Nausea;
        const fatigue = vitals.Fatigue;
        const anemia = vitals.Anemia || vitals['Anemia (baseline)'];

        // Extraire toutes les variables Kræft (cancer) avec valeur = 1
        const cancerVariables = Object.entries(vitals)
            .filter(([key, value]) => 
                (key.toLowerCase().includes('kræft') || 
                 key.toLowerCase().includes('cancer') ||
                 key.toLowerCase().includes('karcinom') ||
                 key.toLowerCase().includes('metastase') ||
                 key.toLowerCase().includes('malign')) && 
                value === 1
            )
            .map(([key]) => key);

        // Construire le résumé en phrases
        const summaryParts = [];

        // Démographie de base
        if (age) {
            summaryParts.push(`${Math.round(Number(age))}-year-old ${gender.toLowerCase()}`);
        } else if (gender) {
            summaryParts.push(`${gender}`);
        }

        // BMI
        if (bmi) {
            const bmiValue = Number(bmi).toFixed(1);
            summaryParts.push(`with a BMI of ${bmiValue}`);
        }

        // Metastasis
        if (metastasis === 1) {
            summaryParts.push(`with metastasis`);
        } else {
            summaryParts.push(`with no metastasis`);
        }

        // COPD
        if (copd === 1) {
            summaryParts.push(`The patient has COPD.`);
        }

        // Variables Kræft (cancer)
        if (cancerVariables.length > 0) {
            const cancerList = cancerVariables.map(cancer => 
                cancer.replace(/\(baseline\)/gi, '').trim()
            ).join(', ');
            summaryParts.push(`Cancer diagnoses: ${cancerList}.`);
        }

        // Symptômes
        const symptoms = [];
        if (pain === 1) symptoms.push('pain');
        if (nausea === 1) symptoms.push('nausea');
        if (fatigue === 1) symptoms.push('fatigue');
        if (anemia === 1) symptoms.push('anemia');

        if (symptoms.length > 0) {
            summaryParts.push(`Symptoms present: ${symptoms.join(', ')}.`);
        }

        // Résultats sanguins
        const bloodWork = [];
        
        if (albumin) {
            const albuminText = albuminDiff ? 
                `Albumin=${Number(albumin).toFixed(1)} (${Number(albuminDiff).toFixed(1)} from previous)` :
                `Albumin=${Number(albumin).toFixed(1)}`;
            bloodWork.push(albuminText);
        }

        if (alkalinePhosphatase) {
            const alkPhosText = alkalinePhosphataseDiff ? 
                `Alkaline phosphatase=${Number(alkalinePhosphatase).toFixed(0)} (${Number(alkalinePhosphataseDiff).toFixed(0)} from previous)` :
                `Alkaline phosphatase=${Number(alkalinePhosphatase).toFixed(0)}`;
            bloodWork.push(alkPhosText);
        }

        if (oxygen) {
            const oxygenText = oxygenDiff ? 
                `Oxygen=${Number(oxygen).toFixed(1)} (${Number(oxygenDiff).toFixed(1)} from previous)` :
                `Oxygen=${Number(oxygen).toFixed(1)}`;
            bloodWork.push(oxygenText);
        }

        if (haemoglobin) {
            const haemoglobinText = haemoglobinDiff ? 
                `Haemoglobin=${Number(haemoglobin).toFixed(1)} (${Number(haemoglobinDiff).toFixed(1)} from previous)` :
                `Haemoglobin=${Number(haemoglobin).toFixed(1)}`;
            bloodWork.push(haemoglobinText);
        }

        if (carbonDioxide) {
            bloodWork.push(`Carbon dioxide=${Number(carbonDioxide).toFixed(1)}`);
        }

        if (bloodWork.length > 0) {
            summaryParts.push(`Blood work: ${bloodWork.join(', ')}.`);
        }

        // Si pas assez d'information, retourner un message générique
        if (summaryParts.length === 0) {
            return "Insufficient data available for a meaningful summary.";
        }

        // Formater le résumé final
        let finalSummary = summaryParts[0];
        
        // Ajouter les parties suivantes avec la bonne ponctuation
        for (let i = 1; i < summaryParts.length; i++) {
            const currentPart = summaryParts[i];
            const previousEndsWithPeriod = summaryParts[i-1].endsWith('.');
            
            if (previousEndsWithPeriod || i === 1) {
                finalSummary += ' ' + currentPart;
            } else {
                finalSummary += ', ' + currentPart;
            }
        }

        // S'assurer que le résumé se termine par un point
        if (!finalSummary.endsWith('.')) {
            finalSummary += '.';
        }

        return finalSummary;
    };

    // function to render the tab content
    const renderTabContent = () => {
        switch (activeTab) {
            case "table":
                return (
                    <div className="patient-table">
                        <div className="table-controls">
                            <div className="zero-filter-toggle">
                                <button 
                                    className={`toggle-button ${showZeroValues ? 'active' : ''}`}
                                    onClick={() => setShowZeroValues(!showZeroValues)}
                                >
                                    {showZeroValues ? '✅' : '❌'} Zero Values: {showZeroValues ? 'Shown' : 'Hidden'}
                                </button>
                            </div>
                        </div>
                        
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
                );

            case "summary":
                return (
                    <div className="summary-tab">
                        <div className="summary-content">
                            <h3>Data Summary</h3>
                            <pre className="summary-text">{generateSummary()}</pre>
                        </div>
                    </div>
                );

            case "ai_summary":
                return (
                    <div className="ai-summary-tab">
                        <div className="ai-summary-content">
                            <h3>🤖 AI Medical Summary</h3>
                            
                            {aiLoading && (
                                <div className="ai-loading">
                                    <Loader />
                                    <p>Generating medical summary with AI...</p>
                                </div>
                            )}
                            
                            {!aiLoading && aiError && (
                                <div className="ai-error">
                                    <p className="error-message">{aiError}</p>
                                    <button 
                                        onClick={generateAISummary}
                                        className="retry-button"
                                    >
                                        Retry
                                    </button>
                                </div>
                            )}
                            
                            {!aiLoading && !aiError && aiSummary && (
                                <>
                                    <div className="summary-text">
                                        {aiSummary}
                                    </div>
                                    {/* og version
                                    <div className="summary-text">
                                        {aiSummary.split('\n').map((line, index) => (
                                            <p key={index} className="summary-line">
                                                {line}
                                            </p>
                                        ))}
                                    </div>
                                    */}
                                    <div className="ai-actions">
                                        <button 
                                            onClick={generateAISummary}
                                            className="regenerate-button"
                                        >
                                            🔄 Regenerate
                                        </button>
                                        <small className="model-info">
                                            Generated by TinyLlama
                                        </small>
                                    </div>
                                </>
                            )}
                            
                            {!aiLoading && !aiError && !aiSummary && (
                                <div className="ai-prompt">
                                    <p>Click the button below to generate an AI-powered medical summary.</p>
                                    <button 
                                        onClick={generateAISummary}
                                        className="generate-button"
                                    >
                                        🚀 Generate AI Summary
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                );


            default:
                return null;
        }
    };

        return (
        <div className="patient-container">
            <div className="patient-overview">
                <div className="patient-header">
                    <h1>Patient {selectedPatientId}: Overview</h1>
                </div>
                
                <div className="patient-search">
                    <SearchBar searchTitle="Vitals" onSearch={setSearchValue} />
                </div>

                {/* Système de Tabs - AJOUTER LE BOUTON AI ICI */}
                <div className="patient-tabs-container">
                    <div className="tabs">
                        <button
                            className={activeTab === "table" ? "tab-active" : "tab"}
                            onClick={() => setActiveTab("table")}
                        >
                            📊 Detailed Table ({filteredVitals.length})
                        </button>
                        <button
                            className={activeTab === "summary" ? "tab-active" : "tab"}
                            onClick={() => setActiveTab("summary")}
                        >
                            📝 Data Summary
                        </button>
                        {/* BOUTON MANQUANT - AJOUTEZ CETTE LIGNE */}
                        <button
                            className={activeTab === "ai_summary" ? "tab-active" : "tab"}
                            onClick={() => setActiveTab("ai_summary")}
                        >
                            🤖 AI Medical Summary
                        </button>
                    </div>

                    <div className="tab-content">
                        {renderTabContent()}
                    </div>
                </div>
            </div>
        </div>
    );
}