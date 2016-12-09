from __future__ import division, print_function
from keras.models import model_from_json
from keras.optimizers import Adamax
import numpy as np
import os

DATA_DIR = "/data/projects/datasets"
DATA_DIR_CHECKPOINTS = "/data/projects/datasets/checkpoints"

# deserialize model
fmods = open(os.path.join(DATA_DIR, "alice_pos_model.json"), "rb")
model_json = fmods.read().decode('UTF-8')
fmods.close()
model = model_from_json(model_json)
model.load_weights(os.path.join(DATA_DIR_CHECKPOINTS, "alice_pos_weights.35-2.04.hdf5"))
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adamax)

# label lookup
label_dict = {}
fpos = open(os.path.join(DATA_DIR, "alice_postags.txt"), "rb")
for line in fpos:
    lid, ltxt, _ = line.strip().decode('UTF-8').split("\t")
    label_dict[int(lid)] = ltxt
fpos.close()

# read ngrams into array
ngram_labels = []
fngrams = open(os.path.join(DATA_DIR, "alice_5grams_pred.txt"), "rb")
for line in fngrams:
    label, ngram = line.strip().decode('UTF-8').split("\t")
    ngram_labels.append((ngram, label))
fngrams.close()

# read word+context vectors and predict from model
fpred = open(os.path.join(DATA_DIR, "alice_test_pred.txt"), "wb")
fvec = open(os.path.join(DATA_DIR, "alice_X_pred.csv"), "rb")
lno = 0
for line in fvec:
    X = np.array([float(x) for x in line.strip().decode('UTF-8').split(",")]).reshape(1, 1500)
    y_ = np.argmax(model.predict(X))
    nl = ngram_labels[lno]
    fpred.write(bytes("%s\t%s\t%s\n" % (nl[0], nl[1], label_dict[y_])), 'UTF-8')
    lno += 1
fvec.close()
fpred.close()