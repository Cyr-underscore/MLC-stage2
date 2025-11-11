export async function fetchPatientVitals(selectedPatientId: string, token: string | null) {
    try {
        const response = await fetch(`http://127.0.0.1:8000/vitals/${selectedPatientId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        const data = await response.json();
        if (data.error) {
            return { error: data.error, vitals: null };
        } else {
            return { error: null, vitals: data.vitals };
        }
    } catch (err) {
        return { error: "Erreur lors de la récupération des données.", vitals: null };
    }
}
