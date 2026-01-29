import { useState, useEffect, useCallback } from 'react';
import { sendMouseTracking } from '../utils/api_mouseTracking';

// Types pour les données de tracking
interface ViewportData {
  width: number;
  height: number;
}

interface PartentElement{
  element: ElementData | null;
} 

interface ElementData {
  tag: string;
  id: string;
  className: string;
  textContent: string;
}

interface MouseEventData {
  x: number;
  y: number;
  timestamp: number;
  url: string;
  path: string;
  viewport: ViewportData;
  element: ElementData;
  eventType: string;
  siblingsCount: number;
}

interface UseMouseTrackingReturn {
  isTracking: boolean;
  startTracking: () => void;
  stopTracking: () => void;
  toggleTracking: () => void;
}

export const useMouseTracking = (
  isEnabled: boolean = false, 
  token: string | null = null
): UseMouseTrackingReturn => {
  const [isTracking, setIsTracking] = useState<boolean>(isEnabled);
  const [lastSent, setLastSent] = useState<number>(0);
  
  // Fonction typée pour envoyer les données
  const sendMouseData = useCallback(async (event: MouseEvent) => {
    const now = Date.now();
    
    // Limiter à 1 envoi toutes les 100ms maximum
    if (now - lastSent < 100) {
      return;
    }
    
    const mouseData: MouseEventData = {
      x: event.clientX,
      y: event.clientY,
      timestamp: now,
      url: window.location.href,
      path: window.location.pathname,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      element: {
        tag: event.target?.tagName || '',
        id: event.target?.id || '',
        className: typeof event.target?.className === 'string' 
          ? event.target.className 
          : ''
      },
      eventType: event.type
    };
    
    setLastSent(now);
    await sendMouseTracking(mouseData, token);
  }, [lastSent, token]);

  useEffect(() => {
    if (!isTracking) return;

    console.log("🔍 [MOUSE TRACKING] Starting mouse tracking");
    
    // Événements à tracker
    const events: (keyof DocumentEventMap)[] = [
      'mousemove', 
      'click', 
      'mousedown', 
      'mouseup'
    ];
    
    events.forEach(eventType => {
      document.addEventListener(eventType, sendMouseData as EventListener);
    });

    // Cleanup
    return () => {
      console.log("🔍 [MOUSE TRACKING] Stopping mouse tracking");
      events.forEach(eventType => {
        document.removeEventListener(eventType, sendMouseData as EventListener);
      });
    };
  }, [isTracking, sendMouseData]);

  return {
    isTracking,
    startTracking: () => setIsTracking(true),
    stopTracking: () => setIsTracking(false),
    toggleTracking: () => setIsTracking(prev => !prev)
  };
};