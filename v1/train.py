from keras.layers import Conv2D, UpSampling2D
from keras.layers import InputLayer
from keras.models import Sequential
from skimage.color import rgb2lab
import numpy as np
import os
import cv2

# Get images
X = []
for filename in os.listdir('/DataSet/OrangOutanSet'):
    img = cv2.imread('/DataSet/OrangOutanSet/'+filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(256,256))
    X.append(img)
X = np.array(X, dtype=float)
Xtrain = rgb2lab(1.0/255*X)[:,:,:,0]
Ytrain = rgb2lab(1.0/255*X)[:,:,:,1:]
Xtrain = Xtrain.reshape(Xtrain.shape+(1,))
Ytrain /= 128 

#Design the neural network
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
        
# Train model
model.compile(optimizer='adam',loss='mse')
model.fit(x=Xtrain, y=Ytrain, epochs=100, batch_size=10)

#Save model
model.save('/output/model.h5')
