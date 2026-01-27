import time
import cv2
import threading
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# --- CONFIGURATION DU PROJET ---
MODEL_TYPE = 'yolov8s.pt'
CONFIDENCE = 0.50
VIDEO_FPS = 60

# --- CONFIGURATION CLIMATIQUE (TOULON) ---
SAISON_ACTUELLE = "ETE"  # Choix: "HIVER", "ETE", "MI_SAISON"

# Paramètres de base
PARAMS = {
    "HIVER":     {"target": 20.0, "mode": "CHAUFFAGE"},
    "ETE":       {"target": 25.0, "mode": "CLIM_FROID"},
    "MI_SAISON": {"target": 22.0, "mode": "VENTILATION"}
}

# Variables partagées
frame_buffer = None
latest_boxes = []
person_count = 0
lock = threading.Lock()
running = True

# Variables de Régulation
current_hvac_power = 0  # Puissance ventilateur (0-100%)
adjusted_target = 0.0   # Nouvelle température cible calculée

# --- FONCTION DE RÉGULATION THERMIQUE ---
def calculate_regulation(n_personnes, saison):
    """
    C'est ici que l'IA décide de la température.
    Règle : 1 personne = ~100 Watts de chaleur ajoutée.
    """
    base_target = PARAMS[saison]["target"]
    mode = PARAMS[saison]["mode"]
    
    thermal_load = n_personnes * 100  # Charge en Watts
    new_target = base_target
    fan_speed = 0 # Vitesse ventilateur 0-100%

    if saison == "ETE":
        # En été, les gens chauffent la salle -> On doit refroidir plus fort.
        # Règle : Pour chaque 1000W (10 pers), on baisse la consigne de 1°C pour compenser
        drop = (n_personnes / 10.0) 
        new_target = base_target - drop
        
        # Le ventilateur accélère s'il y a du monde
        if n_personnes > 0:
            fan_speed = 20 + (n_personnes * 5) # Min 20%, +5% par personne
        else:
            fan_speed = 10 # Mode Eco

    elif saison == "HIVER":
        # En hiver, les gens chauffent la salle -> On peut BAISSER le radiateur (économie !)
        # Règle : Pour chaque 10 personnes, on baisse le chauffage de 1°C
        drop = (n_personnes / 10.0)
        new_target = base_target - drop
        fan_speed = 10 # Basse vitesse pour ne pas créer de courant d'air froid

    # Bornes de sécurité (pour ne pas geler ou cuire les gens)
    fan_speed = min(100, max(0, fan_speed))
    
    return new_target, fan_speed, thermal_load

# --- THREAD IA (VISION) ---
def ai_worker():
    global latest_boxes, person_count, frame_buffer
    print(f"Chargement du modèle {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE)
    print("IA Prête.")
   
    while running:
        img_for_ai = None
        with lock:
            if frame_buffer is not None:
                img_for_ai = frame_buffer.copy()
       
        if img_for_ai is not None:
            results = model(img_for_ai, classes=[0], conf=CONFIDENCE, verbose=False)
            temp_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    temp_boxes.append((int(x1), int(y1), int(x2), int(y2)))
           
            with lock:
                latest_boxes = temp_boxes
                person_count = len(temp_boxes)
        else:
            time.sleep(0.01)

# --- PROGRAMME PRINCIPAL ---
def main():
    global frame_buffer, running, adjusted_target, current_hvac_power
   
    print(f"--- SMART COMFORT PI : Démarrage (Mode {SAISON_ACTUELLE}) ---")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    WINDOW_NAME = "Smart Comfort Dashboard"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    t = threading.Thread(target=ai_worker)
    t.daemon = True
    t.start()

    last_regulation_time = 0
    
    try:
        while True:
            # 1. Capture & Correction
            frame_rgb = picam2.capture_array()
            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            with lock:
                frame_buffer = frame_rgb
                boxes_to_draw = list(latest_boxes)
                count_now = person_count

            # 2. RÉGULATION (Calcul toutes les 1 seconde pour ne pas spammer)
            if time.time() - last_regulation_time > 1.0:
                target, speed, load = calculate_regulation(count_now, SAISON_ACTUELLE)
                adjusted_target = round(target, 1)
                current_hvac_power = int(speed)
                last_regulation_time = time.time()
                
                # Simulation d'envoi vers l'interface CVC (Risk R1)
                # print(f"[CVC LINK] Envoi commande: Consigne={adjusted_target}°C, Fan={current_hvac_power}%")

            # 3. Dessin des boîtes
            for (x1, y1, x2, y2) in boxes_to_draw:
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 4. DASHBOARD - Overlay Visuel
            # Fond du panneau
            cv2.rectangle(frame_display, (0, 0), (640, 90), (30, 30, 30), -1)
            
            # Colonne 1 : Détection
            cv2.putText(frame_display, f"Occupants: {count_now}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_display, f"Apport: +{count_now * 100} W", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

            # Colonne 2 : Décision IA
            cv2.putText(frame_display, f"Saison: {SAISON_ACTUELLE}", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame_display, f"Mode: {PARAMS[SAISON_ACTUELLE]['mode']}", (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Colonne 3 : Consigne Clim (Le résultat final)
            color_temp = (0, 0, 255) if SAISON_ACTUELLE == "ETE" else (255, 100, 0)
            cv2.putText(frame_display, f"CIBLE: {adjusted_target} C", (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_temp, 2)
            cv2.putText(frame_display, f"FAN: {current_hvac_power}%", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

    except KeyboardInterrupt:
        running = False
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        t.join(timeout=1)

if __name__ == "__main__":
    main()