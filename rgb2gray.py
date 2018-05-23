import cv2
import os

X = []

for filename in os.listdir('dirname1/'):
    img = cv2.imread('dirname1/'+filename, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    X.append(img)
    
for i in range(len(X)):
    cv2.imwrite('dirname2/img_'+str(i)+'.png', X[i])
    