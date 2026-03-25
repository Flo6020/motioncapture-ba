import cv2
import numpy as np

print("OpenCV-Version:", cv2.__version__)

# Ausgabe des aktuell verwendeten GUI-Backends von OpenCV
# z. B. QT, GTK oder WAYLAND
# wichtig zur Diagnose von Anzeigeproblemen unter Linux/WSL
print("UI-Backend:", cv2.currentUIFramework())

# Erzeugung eines schwarzen Testbildes mit der Größe 480x640 Pixel
# Format: Höhe x Breite x Farbkanäle
img = np.zeros((480, 640, 3), dtype=np.uint8)

#Text
cv2.putText(
    img,                        # Zielbild
    "OpenCV GUI Test",          # Textinhalt
    (120, 240),                 # Position (x, y)
    cv2.FONT_HERSHEY_SIMPLEX,   # Schriftart
    1,                          # Schriftgröße
    (255, 255, 255),            # Textfarbe (weiß)
    2                           # Linienstärke
)

# Öffnet Fenster
cv2.imshow("test", img)

# Parameter 0: unendlich lange waren
cv2.waitKey(0)
cv2.destroyAllWindows()