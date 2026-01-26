import time
import cv2
import threading
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_TYPE = 'yolov8s.pt'  # Modèle Small (Précis mais lent)
CONFIDENCE = 0.60          # 60% de confiance minimum
VIDEO_FPS = 60             # Objectif fluidité vidéo

# Variables partagées entre les threads
frame_buffer = None        # La dernière image capturée
latest_boxes = []          # Les derniers carrés détectés
person_count = 0
lock = threading.Lock()    # Sécurité pour éviter les conflits
running = True

# --- THREAD IA (L'ANALYSE EN ARRIÈRE-PLAN) ---
def ai_worker():
    global latest_boxes, person_count, frame_buffer
   
    print(f"Chargement du modèle {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE)
    print("IA Prête et en attente.")
   
    while running:
        img_for_ai = None
       
        # On récupère la dernière image disponible
        with lock:
            if frame_buffer is not None:
                img_for_ai = frame_buffer.copy()
       
        if img_for_ai is not None:
            # L'IA travaille ici (ça prendra ~200ms)
            # On force verbose=False pour gagner un peu de temps
            results = model(img_for_ai, classes=[0], conf=CONFIDENCE, verbose=False)
           
            temp_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    # On stocke les coordonnées
                    temp_boxes.append((int(x1), int(y1), int(x2), int(y2)))
           
            # Mise à jour des résultats pour l'affichage
            with lock:
                latest_boxes = temp_boxes
                person_count = len(temp_boxes)
        else:
            time.sleep(0.01)

# --- PROGRAMME PRINCIPAL (AFFICHAGE VIDÉO) ---
def main():
    global frame_buffer, running
   
    print("Démarrage Caméra...")
    picam2 = Picamera2()
   
    # ON DEMANDE DU RGB PUR pour corriger le problème des têtes bleues
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    WINDOW_NAME = "Smart Energy - 60FPS Display"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Lancer l'IA en parallèle
    t = threading.Thread(target=ai_worker)
    t.daemon = True
    t.start()

    print("Système lancé. Vidéo fluide, IA en fond.")

    try:
        while True:
            # 1. Capture ultra rapide (RGB)
            frame_rgb = picam2.capture_array()

            # 2. Correction Couleur (RGB -> BGR pour OpenCV)
            # C'est CETTE ligne qui va régler ton problème de têtes bleues
            frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 3. Envoi à l'IA (on envoie l'image propre)
            with lock:
                frame_buffer = frame_rgb # L'IA préfère le RGB souvent

            # 4. Récupération des dessins (sans attendre l'IA)
            boxes_to_draw = []
            count_to_show = 0
            with lock:
                boxes_to_draw = list(latest_boxes)
                count_to_show = person_count

            # 5. Dessin manuel (Plus joli et plus rapide que r.plot())
            for (x1, y1, x2, y2) in boxes_to_draw:
                # Carré VERT (0, 255, 0) - Change ici si tu veux une autre couleur
                # (Rappel: en OpenCV c'est Blue, Green, Red)
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_display, "Personne", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 6. Infos Textuelles
            cv2.rectangle(frame_display, (0, 0), (300, 50), (0, 0, 0), -1)
            cv2.putText(frame_display, f"Pers: {count_to_show}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 7. Affichage
            cv2.imshow(WINDOW_NAME, frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

    except KeyboardInterrupt:
        running = False
        print("Arrêt...")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        t.join(timeout=1)

if __name__ == "__main__":
    main()