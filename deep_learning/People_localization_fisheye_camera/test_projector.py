
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from projector import Projector

    
if __name__ == '__main__':
    
    coeffs = [0.10535863449258721, -0.03506148489575732, 0.026807731003823377, -0.006314303315952602]  # k1, k2, ..., k4 camera 115
    intrinsics = [4909, 4931]  # fx, fy camera 115
    image_center = [956.44625105, 970.54495143] # camera 155
    
    projector = Projector(coeffs, intrinsics, image_center)
    Z = 2.95
    projector.set_Z(Z)

    imw = 1920 #camera 115
    imh = 1920 # camera 115
    
    camera = Projector(coeffs, intrinsics, image_center)

    # === Génération des valeurs u et v (autour du centre image) ===
    u_vals = np.linspace(camera.u0 - 500, camera.u0 + 500, 100)
    v_vals = np.linspace(camera.v0 - 500, camera.v0 + 500, 100)

    # === Tracer X(u) ===
    X_vals = []
    valid_u = []

    for u in u_vals:
        result = camera.image_to_3d(u, camera.v0, Z)
        if result is not None:
            X_vals.append(result[0])
            valid_u.append(u)

    # === Tracer Y(v) ===
    Y_vals = []
    valid_v = []

    for v in v_vals:
        result = camera.image_to_3d(camera.u0, v, Z)
        if result is not None:
            Y_vals.append(result[1])
            valid_v.append(v)

    # === Affichage ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(valid_u, X_vals, color='blue', marker='o')
    axs[0].axvline(camera.u0, color='red', linestyle='--', label='u₀ (centre image)')
    axs[0].set_xlabel("u (pixels)")
    axs[0].set_ylabel("X (meter)")
    axs[0].set_title("Curve X(u)")
    axs[0].grid(True)

    axs[1].plot(valid_v, Y_vals, color='green', marker='o')
    axs[1].axvline(camera.v0, color='red', linestyle='--', label='u₀ (centre image)')
    axs[1].set_xlabel("v (pixels)")
    axs[1].set_ylabel("Y (meter))")
    axs[1].set_title("Curve Y(v)")
    axs[1].grid(True)

    plt.suptitle("Projection 3D (model fisheye Kannala-Brandt)")
    plt.tight_layout()
    plt.show()
    plt.show(block=True) 


    # *********** Test few points *********** #
    projector.verbose = True
    print("* Point cohérent *")
    projector(700, 700)
    print("* En haut à gauche *")
    projector(0, 0)
    print("* Au milieu *")
    projector(1920 // 2, 1920 // 2)
    print(" En bas à droite *")
    projector(1920, 1920)
    projector(1920, 1920)
    print(" point réel A *")
    projector(517*1920/1024, 400*1920/1024)
    print(" point réel B *")
    projector(643*1920/1024, 403*1920/1024)
    print(" point réel C *")
    projector(702*1920/1024, 524*1920/1024)
    print(" point réel D *")
    projector(700*1920/1024, 583*1920/1024)
    print(" point réel E *")
    projector(595*1920/1024, 569*1920/1024)
    print(" point réel F *")
    projector(439*1920/1024, 436*1920/1024)
    print(" point réel G *")
    projector(409*1920/1024, 495*1920/1024)
    print(" point réel H1 *")
    projector(686*1920/1024, 534*1920/1024)
    print(" point réel H2 *")
    projector(633*1920/1024, 641*1920/1024)
    print(" point réel I *")
    projector(708*1920/1024, 387*1920/1024)
    
    # *********** Test with varying r *********** #
    step = 1
    projector.verbose = False
    hm = np.zeros((imw // 2, imh // 2, 3), dtype=np.uint8)
    Xs = []
    Ys = []
    ratio = 255 / 5.0
    for ir_int in range(0, 5000, step):
        ir = ir_int / 10000
        # for v in range(0, imh, step):
        if True:
            phi = math.radians(-90)

            # ir = abs(ip - 0.5)
            r = ir / 4
            # print(f"r: {r}")
            # print(f"u: {u}, v: {v}")
            X, Y = projector.get_loc(r, phi)
            if X is None:
                continue

            Xs.append(X)
            Ys.append(Y)

            r_pxl = int(ir * imw)
            u = r_pxl
            v = r_pxl
            hm[v, u, :] = np.uint8(np.sqrt(X**2 + Y**2)*ratio)

    # hm = cv2.resize(hm, (640, 640))
    # cv2.imshow('hm', hm)
    # cv2.waitKey(0)
    
    #exit()

    # *********** Test on the all image *********** #
    step = 1
    projector.verbose = False
    hm = np.zeros((imw, imh, 3), dtype=np.uint8)
    ratio = 255 / 17.
    for u in range(0, imw, step):
        for v in range(0, imh, step):
            # print(f"u: {u}, v: {v}")
            X, Y = projector(u, v)
            if X is None:
                continue

            hm[v, u, :] = np.uint8(np.sqrt(X**2 + Y**2)*ratio)
       
     # Affichage des points de test sur la carte de chaleur
            
    scale_point = 1920 / 1024
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (0, 255, 0)
    thickness = 2
    
    # === Génération de la carte de distance ===
    hm = np.zeros((imh, imw, 3), dtype=np.uint8)
    distance_map = np.zeros((imh, imw), dtype=np.float32)
    max_distance = 10 
    ratio = 255. / max_distance

    for u in range(0, imw, step):
        for v in range(0, imh, step):
            X, Y = projector(u, v)
            if X is None:
                continue
            d = np.sqrt(X**2 + Y**2)
            intensity = int(min(d * ratio, 255))
            hm[v, u, :] = (intensity, 0, 255 - intensity)  # Colormap simple
            distance_map[v, u] = d
            
##################################################################################################################""
    # Étape 1 : Normaliser la distance pour colormap OpenCV (8-bit)
    distance_map_norm = np.clip((distance_map / max_distance) * 255, 0, 255).astype(np.uint8)

    # Appliquer la colormap  d’OpenCV
    hm_color = cv2.applyColorMap(distance_map_norm, cv2.COLORMAP_INFERNO )

    #  Ajouter cercles isodistants avec OpenCV
    center = (imw // 2, imh // 2)
    distances_m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    color_circle = (255, 255, 255)
    for d_m in distances_m:
        radius = int((d_m / max_distance) * (imw // 2))  # Proportionnel à l’image
        cv2.circle(hm_color, center, radius, color_circle, thickness=1)

    # Ajouter les points d'intérêt
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (0, 255, 0)
    thickness = 2

    points = {
        "HG": (5, 5),
        "camera": (1920 // 2, 1920 // 2),
        "BD": (1900, 1900),
        "A": (int(517 * scale_point), int(400 * scale_point)),
        "B": (int(643 * scale_point), int(403 * scale_point)),
        "C": (int(702 * scale_point), int(524 * scale_point)),
        "D": (int(700 * scale_point), int(583 * scale_point)),
        "E": (int(595 * scale_point), int(569 * scale_point)),
        "F": (int(439 * scale_point), int(436 * scale_point)),
        "G": (int(409 * scale_point), int(495 * scale_point)),
        "H1": (int(686 * scale_point), int(534 * scale_point)),
        "H2": (int(633 * scale_point), int(641 * scale_point)),
        "I": (int(708 * scale_point), int(387 * scale_point)),
        
    }

    for name, (x, y) in points.items():
        cv2.circle(hm_color, (x, y), radius=8, color=color, thickness=-1)
        cv2.putText(hm_color, name, (x + 10, y - 10), font, font_scale, color, thickness, cv2.LINE_AA)

    # Affichage avec matplotlib (et colorbar correcte)
    hm_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    distance_map_resized = cv2.resize(distance_map, (920, 920), interpolation=cv2.INTER_LINEAR)
    hm_rgb_resized = cv2.resize(hm_rgb, (920, 920), interpolation=cv2.INTER_LINEAR)

# ############################################################################################

    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    im = ax.imshow(hm_rgb_resized)
    plt.title("Radial distance heat map with isodist circles")
    plt.axis('off')

    # Créer un mappable pour la colorbar
    norm = Normalize(vmin=0, vmax=max_distance)
    sm = ScalarMappable(norm=norm, cmap='inferno')
    sm.set_array([])

    # Ajouter la colorbar avec association explicite à ax
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Distance to center (meters)")
    cb.set_ticks(np.linspace(0, max_distance, 6))
    cb.set_ticklabels([f"{i:.1f}" for i in np.linspace(0, max_distance, 6)])

    plt.tight_layout()
    plt.show()
    
#################################################################

    # *********** Test some points with a hight step *********** #
    projector.verbose = True
    step = 100
    for u in range(0, imw, step):
        # for v in range(0, imh, step):
        if True:
            v = u
            print(f"u: {u}, v: {v}")
            X, Y = projector(u, v)
            if X is None:
                continue

