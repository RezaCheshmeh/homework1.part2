# homework1.part2
تمرین یک بخش دو
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,  Activation
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
np.unique(train_labels)
np.max(train_labels)+1
for i in range(10):
  plt.imshow(train_images[i])
  plt.figure()
x_train = train-images.astype('float32')
x_test = test_images.astype('float32')
train_images=[image.reshape(28*28) for image in train_images]
test_images=[image.reshape(28*28) for image in test_images]
x_train = np.array(train_images)
x_test = np.array(test_images)
# 4. Preprocess class labels
y_train = keras.utils.to_categorical(train_labels, num_classes=10)
y_test = keras.utils.to_categorical(test_labels, num_classes=10)

# 5. Define model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=28*28))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 6. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
              epochs=200,
              batch_size=64, validation_split=0.2)
