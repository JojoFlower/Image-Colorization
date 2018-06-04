import cv2
import os
import sys

img = cv2.imread(sys.argv[1], 0)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.imwrite('img.png', img)
