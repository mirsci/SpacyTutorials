from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adamax
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "/data/projects/datasets"

np.random.seed(42)

# read data
X = np.loadtxt(os.path.join(DATA_DIR, "alice_X.csv"), delimiter=",")
y = np.loadtxt(os.path.join(DATA_DIR, "alice_y.csv"), delimiter=",")

# set up model
model = Sequential([
    # input layer
    Dense(768, input_shape=(1500,), W_regularizer=l2(0.001)),
    Activation("relu"),
    Dropout(0.2),
    # hidden layer
    Dense(512, W_regularizer=l2(0.001)),
    Activation("relu"),
    Dropout(0.2),
    # output layer
    Dense(13),
    Activation("softmax")
])

adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adamax)

# save model structure
model_struct = model.to_json()
fmod_struct = open(os.path.join(DATA_DIR, "alice_pos_model.json"), "wb")
fmod_struct.write(bytes(model_struct, 'UTF-8'))
fmod_struct.close()

# train model
checkpoint = ModelCheckpoint(os.path.join(DATA_DIR, "checkpoints",
    "alice_pos_weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
    monitor="val_loss", save_best_only=True, mode="min")
model.summary()
print('y shape: ', y.shape)
# y = y.reshape((-1, 1))
# print('y shape after reshaping: ', y.shape)

hist = model.fit(X, y, batch_size=128, nb_epoch=50, verbose=1, shuffle=True,
    validation_split=0.3, callbacks=[checkpoint])

# plot losses
train_loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
plt.plot(range(len(train_loss)), train_loss, color="red", label="Train Loss")
plt.plot(range(len(train_loss)), val_loss, color="blue", label="Val Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="best")
plt.show()