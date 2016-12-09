from gensim.models import word2vec
import numpy as np

print("Loading label lookup...")
label_lookup = {}
f_postags = open("/data/projects/datasets/alice_postags.txt", "rb")
for line in f_postags:
    lid, ltext, _ = line.strip().decode('UTF-8').split("\t")
    label_lookup[ltext] = int(lid)
f_postags.close()

print("Loading word2vec model...")
w2v = word2vec.Word2Vec.load_word2vec_format(
        "/data/projects/datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)
vec_size = 300
vec_pad = np.zeros(vec_size)
vec_unk = np.ones(vec_size)
ngram_size = 5

print("Writing vectors...")
f_data = open("/data/projects/datasets/alice_5grams.txt", "rb")
f_X = open("/data/projects/datasets/alice_X.csv", "wb")
f_y = open("/data/projects/datasets/alice_y.csv", "wb")
nbr_read = 0
for line in f_data:
    nbr_read += 1
    if nbr_read % 1000 == 0:
        print("    Wrote %d vectors..." % (nbr_read))
    label, ngram = line.strip().decode('UTF-8').split("\t")
    lid = label_lookup[label]
    word_vecs = np.zeros((ngram_size, vec_size))
    for i, word in enumerate(ngram.split(" ")):
        if word == "PAD":
            word_vecs[i] = vec_pad
        elif word == "UNK":
            word_vecs[i] = vec_unk
        else:
            try:
                word_vecs[i] = w2v[word]
            except KeyError:
                word_vecs[i] = vec_unk
    ngram_vec = np.reshape(word_vecs, (ngram_size * vec_size))
    f_X.write(bytes("%s\n" % (",".join(["%.5f" % (x) for x in ngram_vec.tolist()])),'UTF-8'))
    label_vec = np.zeros(len(label_lookup))
    label_vec[lid] = 1
    f_y.write(bytes ("%s\n" % (",".join(["%d" % (x) for x in label_vec.tolist()])),'UTF-8'))
print("Wrote %d vectors" % (nbr_read))
f_X.close()
f_y.close()