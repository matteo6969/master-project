import cv2
from ultralytics import YOLO

# 1. Chargement du modèle
# On utilise 'yolov8n.pt' (nano) qui est léger et rapide. 
# Au premier lancement, il va se télécharger automatiquement.
print("Chargement du modèle IA...")
model = YOLO('yolov8n.pt')

# 2. Ouverture de la webcam (0 est généralement la webcam intégrée)
cap = cv2.VideoCapture(0)

# On règle la résolution pour que ce soit fluide (640x480 est standard pour l'IA)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

print("Démarrage... Appuie sur 'q' pour quitter.")

while True:
    # Lecture de l'image de la webcam
    success, img = cap.read()
    if not success:
        break

    # 3. Détection IA
    # stream=True rend l'inférence un peu plus fluide
    # classes=[0] signifie qu'on ne veut détecter QUE les personnes (0 = personne dans COCO dataset)
    results = model(img, stream=True, classes=[0], verbose=False)

    person_count = 0

    # 4. Traitement des résultats
    for r in results:
        # Création de l'image annotée (avec les cadres autour des gens)
        img_annotated = r.plot()
        
        # Comptage des boîtes détectées
        person_count = len(r.boxes)

    # 5. Affichage du nombre de personnes en gros sur l'écran
    # C'est cette valeur qui servira plus tard à régler la clim
    cv2.putText(img_annotated, f"Personnes: {person_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage de la fenêtre
    cv2.imshow("Test IA - Detection Personnes", img_annotated)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage à la fin
cap.release()
cv2.destroyAllWindows()