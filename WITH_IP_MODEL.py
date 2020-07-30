import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, AveragePooling1D, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
import gc
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

X = np.load('Sequences_Random_50.npy', allow_pickle=True)
print("Data Read")
print(np.shape(X))
y = np.load('Labels_Random_50.npy', allow_pickle=True)[1:]
y = y[:, 0]
print("Labels Read")
countDDoS = 0
for i in y:
  if i == 1:
    countDDoS+=1
print("Total DDos %: " + str(countDDoS/len(y)))
X, y = shuffle(X, y, random_state=19)
X_train, X_val, X_test = np.split(X, [int(.8*len(X)), int(.9*len(X))])
y_train, y_val, y_test = np.split(y, [int(.8*len(y)), int(.9*len(y))])

del X
del y
gc.collect()

countDDoS = 0
for i in y_train:
  if i == 1:
    countDDoS+=1
print("Train DDos %: " + str(countDDoS/len(y_train)))

countDDoS = 0
for i in y_val:
  if i == 1:
    countDDoS+=1
print("Val DDos %: " + str(countDDoS/len(y_val)))

countDDoS = 0
for i in y_test:
  if i == 1:
    countDDoS+=1
print("Test DDos %: " + str(countDDoS/len(y_test)))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
print("GPU Registered")
gc.collect()
print("Building model....")
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(50,len(X_train[0][0]))))
model.add(GlobalMaxPool1D())
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(16, activation='sigmoid'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.Adam()
print('compileing...')
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy()])
print('compiled')
print('Training')
history = model.fit(X_train,y_train,verbose=1,epochs=10, shuffle=False, validation_data=(X_val, y_val))

print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('Accuracy per Epoch With IP.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('Loss per Epoch With IP.png')

print(model.evaluate(X_test, y_test))
print(model.evaluate(X_val, y_val))

predictions = model.predict(X_val)
#print(tf.math.confusion_matrix(y_val, predictions, num_classes=2))
print(confusion_matrix(y_val, predictions.round()))
print(f1_score(y_val, predictions.round(), average='macro'))


predictions = model.predict(X_train)
#print(tf.math.confusion_matrix(y_train, predictions, num_classes=2))
print(confusion_matrix(y_train, predictions.round()))
print(f1_score(y_train, predictions.round(), average='macro'))

predictions = model.predict(X_test)
#print(tf.math.confusion_matrix(y_test, predictions, num_classes=2))
print(confusion_matrix(y_test, predictions.round()))
print(f1_score(y_test, predictions.round(), average='macro'))

model.save('WITH_IP_MODEL')