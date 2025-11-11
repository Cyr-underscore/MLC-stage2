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

export async function fetchPatients({
    page,
    limit,
    searchIdFilter,
    ageMinFilter,
    ageMaxFilter,
    token
}: {
    page: number,
    limit: number,
    searchIdFilter?: string,
    ageMinFilter?: string,
    ageMaxFilter?: string,
    token: string | null
}) {
    const skip = page * limit;
    const queryParams = new URLSearchParams({
        skip: skip.toString(),
        limit: limit.toString(),
    });
    if (searchIdFilter) queryParams.append('id_patient', searchIdFilter);
    if (ageMinFilter) queryParams.append('age_min', ageMinFilter);
    if (ageMaxFilter) queryParams.append('age_max', ageMaxFilter);

    try {
        const response = await fetch(`http://localhost:8000/patients?${queryParams.toString()}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        const data = await response.json();
        return data;
    } catch (err) {
        return { patients: [], total: 0 };
    }
}
