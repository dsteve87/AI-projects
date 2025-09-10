import cv2
import os
import time

"""
    ce code permet de capturer des images chaque 5 secondes à partir d'un ou plusieurs flux vidéo
"""
# --- Paramètres utilisateur ---
camera_ids = {
    
    "Cam_002": {
        "url": " ",
        "prefix": "im125"
    },
    "Cam_003": {
        "url": " ",
        "prefix": "im115"
    },
    
    "Cam_001": {
        "url": " ",
        "prefix": "im122"
    }
}
output_dir = " "
image_format = ".png"
max_images = 150
capture_interval = 5 # secondes

# --- Créer les sous-dossiers pour chaque caméra ---
for cam_name in camera_ids:
    os.makedirs(os.path.join(output_dir, cam_name), exist_ok=True)

# --- Initialiser les captures vidéo ---
caps = {}
for cam_name, info in camera_ids.items():
    cap = cv2.VideoCapture(info["url"])
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la caméra {cam_name} ({info['url']})")
        exit()
    caps[cam_name] = cap

print(f"Captures toutes les {capture_interval} secondes, jusqu'à {max_images} images. Appuie sur ÉCHAP pour quitter.")

img_count = 0
last_capture_time = time.time()

while img_count < max_images:
    frames = {}
    for cam_name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            print(f"Échec de lecture de la frame pour {cam_name}")
            continue
        # Affiche compteur sur l’image
        cv2.putText(frame, f"{cam_name} Captures: {img_count}/{max_images}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 7)
        cv2.imshow(cam_name, frame)
        frames[cam_name] = frame

    key = cv2.waitKey(1)
    if key == 27:  # ÉCHAP
        print("Arrêt par l'utilisateur.")
        break

    # Capture toutes les 5 secondes
    if time.time() - last_capture_time >= capture_interval:
        for cam_name, frame in frames.items():
            prefix = camera_ids[cam_name]["prefix"]
            filename = os.path.join(output_dir, cam_name, f"{prefix}_{img_count:02d}{image_format}")
            cv2.imwrite(filename, frame)
            print(f"[{cam_name}] Image enregistrée : {filename}")
        img_count += 1
        last_capture_time = time.time()
    

# --- Nettoyage ---
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
