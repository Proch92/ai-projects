from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np

model = Sequential()

model.add(Dense(3, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)

model.compile(loss='binary_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

x_train = np.zeros((4,2))
y_train = np.zeros((4,1))
for x in [0,1]:
      for y in [0,1]:
          x_train[(x*2)+y] = [x,y]
          y_train[(x*2)+y] = [x^y]

print(x_train)
print(y_train)

model.fit(x_train, y_train, epochs=500, batch_size=1)

x_test = np.array([[1,0],[1,1],[0,1],[0,0]])
classes = model.predict(x_test)
print(classes)
