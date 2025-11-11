export function logUserAction(action: string, details: object = {}) {
    const username = localStorage.getItem("username");
    const token = localStorage.getItem("token");

    if (!username || !token) return;

    const defaultDetails = {
        appVersion: "enriched"
    }

    const mergedDetails = { ...defaultDetails, ...details };

    fetch("http://localhost:8000/log_action", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
            username,
            action,
            timestamp: new Date().toISOString(),
            details: mergedDetails,
        }),
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error("Log error:", data.error);
            }
        })
        .catch(err => console.error("Logging failed:", err));
}
