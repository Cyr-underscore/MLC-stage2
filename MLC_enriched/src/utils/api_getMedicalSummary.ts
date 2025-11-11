export async function fetchMedicalSummary(patientId: string, token: string | null) {
    try {
        console.log("🔍 [FETCH DEBUG] Calling API for patient:", patientId);
        
        const response = await fetch(`http://127.0.0.1:8000/medical-summary/${patientId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        console.log("🔍 [FETCH DEBUG] Response status:", response.status);
        
        const data = await response.json();
        console.log("🔍 [FETCH DEBUG] Full API response:", data);
        
        // CORRECTION ICI : Utilisez la bonne clé
        if (data.error) {
            console.log("🔍 [FETCH DEBUG] API returned error:", data.error);
            return { error: data.error, summary: null };
        } else {
            // Essayez différentes clés possibles
            const summary = data.summary || data.medical_summary || data.content || data.result || data;
            console.log("🔍 [FETCH DEBUG] Extracted summary:", summary);
            return { error: null, summary: summary };
        }
    } catch (err) {
        console.error("🔍 [FETCH DEBUG] Catch error:", err);
        return { error: "Error generating medical summary", summary: null };
    }
}