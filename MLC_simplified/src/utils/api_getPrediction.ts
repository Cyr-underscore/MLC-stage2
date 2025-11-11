export async function fetchShapValues(selectedPatientId: string, token: string | null) {
    try {
        const response = await fetch(`http://127.0.0.1:8000/shap/${selectedPatientId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        if (!response.ok) throw new Error("Error API");
        const data = await response.json();
        return { shap_values: data.shap_values, error: null };
    } catch (err) {
        return { shap_values: null, error: `Error fetching SHAP data: ${err}` };
    }
}

export async function fetchPrediction(selectedPatientId: string, token: string | null) {
    try {
        const response = await fetch(`http://localhost:8000/predict/${selectedPatientId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        const data = await response.json();
        return { prediction_proba: data.prediction_proba };
    } catch (err) {
        return { prediction_proba: null };
    }
}
