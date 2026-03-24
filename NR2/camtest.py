import cv2
import numpy as np

print("OpenCV-Version:", cv2.__version__)
print("UI-Backend:", cv2.currentUIFramework())

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, "OpenCV GUI Test", (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()