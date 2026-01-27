import time
import cv2
import threading
import board
import datetime
import adafruit_dht
from picamera2 import Picamera2
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_TYPE = 'yolov8s.pt'
CONFIDENCE = 0.50
DHT_PIN = board.D4

# MARGE (Le "Tunnel")
# Si on vise 20°C avec une marge de 0.5 :
# On allume à 19.5°C et on coupe à 20.5°C
MARGE = 0.5 

# --- CONSTANTES CIBLES (L'Idéal) ---
# ETE (On refroidit)
ETE_VIDE = 29.0    
ETE_CONFORT = 25.0 
ETE_FOULE = 23.0   

# HIVER (On chauffe)
HIVER_VIDE = 15.0    
HIVER_CONFORT = 19.0 
HIVER_FOULE = 17.0   

# Variables partagées
frame_buffer = None
latest_boxes = []
person_count = 0
current_temp = None
current_hum = None
lock = threading.Lock()
running = True

# ==========================================
# 2. LOGIQUE INTELLIGENTE
# ==========================================
def get_saison_automatique():
    mois = datetime.date.today().month
    if 5 <= mois <= 9: return "ETE"
    else: return "HIVER"

def calculer_seuils(nb_personnes, saison):
    """
    Retourne 3 valeurs : Cible, Seuil Allumage, Seuil Extinction
    """
    cible = 0.0
    fan = "OFF"
    etat_ia = ""
    color = (255, 255, 255)

    # 1. D'ABORD ON TROUVE LA CIBLE
    if saison == "ETE":
        if nb_personnes == 0:
            cible = ETE_VIDE
            fan = "ECO"
            etat_ia = "VEILLE"
            color = (0, 255, 0)
        elif nb_personnes < 5:
            cible = ETE_CONFORT
            fan = "MOYEN"
            etat_ia = "CONFORT"
            color = (0, 255, 255)
        elif nb_personnes < 10:
            cible = 24.0
            fan = "FORT"
            etat_ia = "COMPENS."
            color = (0, 165, 255)
        else:
            cible = ETE_FOULE
            fan = "MAX"
            etat_ia = "FOULE"
            color = (0, 0, 255)
        
        # EN ETE (CLIM) : 
        # On allume si T > Cible + Marge (Il fait trop chaud)
        # On coupe si T < Cible - Marge (Il fait assez frais)
        seuil_allumage = cible + MARGE
        seuil_extinction = cible - MARGE

    elif saison == "HIVER":
        if nb_personnes == 0:
            cible = HIVER_VIDE
            fan = "ECO"
            etat_ia = "VEILLE"
            color = (0, 255, 0)
        elif nb_personnes < 5:
            cible = HIVER_CONFORT
            fan = "MOYEN"
            etat_ia = "CONFORT"
            color = (0, 255, 255)
        elif nb_personnes < 10:
            cible = 18.0
            fan = "BAS"
            etat_ia = "COMPENS."
            color = (0, 165, 255)
        else:
            cible = HIVER_FOULE
            fan = "MIN"
            etat_ia = "ARRET"
            color = (0, 0, 255)

        # EN HIVER (CHAUFFAGE) :
        # On allume si T < Cible - Marge (Il fait trop froid)
        # On coupe si T > Cible + Marge (Il fait assez chaud -> C'est ton Seuil Max)
        seuil_allumage = cible - MARGE
        seuil_extinction = cible + MARGE
            
    return cible, seuil_allumage, seuil_extinction, fan, etat_ia, color

# ==========================================
# 3. WORKERS (TACHES DE FOND)
# ==========================================
def ai_worker():
    global latest_boxes, person_count, frame_buffer
    model = YOLO(MODEL_TYPE)
    while running:
        img_copy = None
        with lock:
            if frame_buffer is not None: img_copy = frame_buffer.copy()
        if img_copy is not None:
            results = model(img_copy, classes=[0], conf=CONFIDENCE, verbose=False)
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

def dht_worker():
    global current_temp, current_hum
    try: dht_device = adafruit_dht.DHT11(DHT_PIN)
    except: return
    while running:
        try:
            t = dht_device.temperature
            h = dht_device.humidity
            if t is not None:
                with lock: current_temp = t; current_hum = h
        except: pass
        time.sleep(2.0)

# ==========================================
# 4. AFFICHAGE ET DECISION FINALE
# ==========================================
def main():
    global frame_buffer, running
    
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    cv2.namedWindow("Smart Dashboard", cv2.WINDOW_NORMAL)

    t1 = threading.Thread(target=ai_worker)
    t2 = threading.Thread(target=dht_worker)
    t1.daemon = True; t2.daemon = True
    t1.start(); t2.start()

    try:
        while True:
            # A. INFO SAISON
            saison = get_saison_automatique()
            # Pour tester l'hiver avec ta pièce à 22°C :
            # saison = "HIVER"

            frame_rgb = picam2.capture_array()
            frame_disp = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            with lock:
                frame_buffer = frame_rgb
                nb = person_count
                temp = current_temp
                boxes = list(latest_boxes)

            # B. CALCUL DES SEUILS
            cible, s_on, s_off, fan, etat, color = calculer_seuils(nb, saison)

            # C. DECISION REELLE (ON/OFF)
            action_txt = "STANDBY"
            action_color = (100, 100, 100) # Gris

            if temp is not None:
                if saison == "HIVER":
                    # Chauffage
                    if temp < s_on:
                        action_txt = "CHAUFFE"
                        action_color = (0, 0, 255) # Rouge
                    elif temp > s_off:
                        action_txt = "STOP (MAX)"
                        action_color = (0, 255, 0) # Vert
                    else:
                        action_txt = "ZONE CONFORT" # Entre les deux
                        action_color = (0, 255, 255)
                
                else: # ETE
                    # Clim
                    if temp > s_on:
                        action_txt = "CLIM ON"
                        action_color = (255, 0, 0) # Bleu
                    elif temp < s_off:
                        action_txt = "STOP (MAX)"
                        action_color = (0, 255, 0) # Vert
                    else:
                        action_txt = "ZONE CONFORT"

            # D. DESSIN DASHBOARD (3 COLONNES)
            # Fond
            cv2.rectangle(frame_disp, (0, 0), (640, 120), (20, 20, 20), -1)

            # 1. REEL
            if temp is not None: t_str = f"{temp:.1f} C"
            else: t_str = "--.- C"
            cv2.putText(frame_disp, "REEL", (10, 20), 0, 0.5, (200, 200, 200), 1)
            cv2.putText(frame_disp, t_str, (10, 55), 0, 1.0, (255, 255, 255), 2)
            cv2.putText(frame_disp, f"{nb} Pers.", (10, 90), 0, 0.7, (0, 255, 0), 2)

            # 2. STRATEGIE (Le Tunnel)
            cv2.putText(frame_disp, "STRATEGIE", (200, 20), 0, 0.5, (200, 200, 200), 1)
            
            # Affichage clair : ON < CIBLE > OFF
            # Seuil BAS (Allumage en hiver)
            cv2.putText(frame_disp, f"ON : {s_on}", (200, 45), 0, 0.5, (100, 255, 255), 1)
            # CIBLE
            cv2.putText(frame_disp, f"BUT: {cible}", (200, 75), 0, 0.9, color, 2)
            # Seuil HAUT (Extinction)
            cv2.putText(frame_disp, f"OFF: {s_off}", (200, 100), 0, 0.5, (100, 255, 100), 1)

            # 3. ACTION
            cv2.putText(frame_disp, f"MODE {saison}", (450, 20), 0, 0.5, (200, 200, 200), 1)
            cv2.putText(frame_disp, action_txt, (420, 60), 0, 0.8, action_color, 2)
            cv2.putText(frame_disp, f"FAN: {fan}", (450, 90), 0, 0.5, (200, 200, 200), 1)

            # Dessin Carrés
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Smart Dashboard", frame_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'): running = False; break

    except KeyboardInterrupt: running = False
    finally: picam2.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()