// Types pour les données de souris
export interface ViewportData {
  width: number;
  height: number;
}

export interface ElementData {
  tag: string;
  id: string;
  className: string;
}

export interface MouseEventData {
  x: number;
  y: number;
  timestamp: number;
  url: string;
  path: string;
  viewport: ViewportData;
  element: ElementData;
  eventType: string;
}

export async function sendMouseTracking(
  mouseData: MouseEventData, 
  token: string | null = null
): Promise<{ error: string | null; success: boolean }> {
  try {
    console.log("🔍 [MOUSE TRACKING] Sending mouse data:", mouseData);
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    // Ajouter le token d'authentification si disponible
    // Le format "Bearer token" est le standard pour JWT/OAuth
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    const response = await fetch('http://127.0.0.1:8000/api/mouse-tracking', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(mouseData)
    });
    
    console.log("🔍 [MOUSE TRACKING] Response status:", response.status);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("🔍 [MOUSE TRACKING] API response:", data);
    
    return { error: null, success: true };
    
  } catch (err) {
    console.error("🔍 [MOUSE TRACKING] Catch error:", err);
    const errorMessage = err instanceof Error ? err.message : "Unknown error";
    return { error: `Error sending mouse tracking data: ${errorMessage}`, success: false };
  }
}