from __future__ import division, print_function
import spacy
import operator
import os

DATA_DIR = "/data/projects/datasets"
# Very important, as the models are loaded in a different location
spacy.util.set_data_path('/data/projects/python3.4/spacy')
from spacy.en import English

nlp = English()

falice = open(os.path.join(DATA_DIR, "alice_in_wonderland.txt"), "rb")
content = falice.read()
falice.close()

doc = nlp(content.decode("utf-8"))
sents = []
for sent in doc.sents:
    sents.append(sent)

i = 0
vocab = {}
tags = {}
tagged_sents = []
SPACE = " "
FORWARD_SLASH = "/"

fout = open(os.path.join(DATA_DIR, "alice_sents_postagged.txt"), "wb")
for sent in sents:
    i += 1
    print("Processing sentence# %d: %s" % (i, sent))
    token_tags = []
    toks = nlp(sent.text.encode("ascii", "ignore").decode("utf-8"))
    for tok in toks:
        token_tags.append((tok.text, tok.pos_))
    clean_token_tags = []
    for tt in token_tags:
        if tt[1] == u"SPACE":
            continue
        clean_token_tags.append(tt)
        vocab[tt[0]] = vocab.get(tt[0], 0) + 1
        tags[tt[1]] = tags.get(tt[1], 0) + 1
    tagged_sents.append(clean_token_tags)
    for tt in clean_token_tags:
        str = SPACE.join(FORWARD_SLASH.join([tt[0], tt[1]]))
        fout.write(bytes("%s\n" % str, 'UTF-8'))
    # fout.write("%s\n" % (" ".join(["/".join([tt[0], tt[1]])
    #     for tt in clean_token_tags]) ))
fout.close()

# replace words which occur 1 or 2 times with UNK in vocab
for word in list(vocab):
    if vocab[word] < 3:
        vocab["UNK"] = vocab.get("UNK", 0) + 1
        vocab.pop(word, None)

# create a lookup dictionary for words
fwords = open(os.path.join(DATA_DIR, "alice_words.txt"), "wb")
vocab_s = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
for i, (k, v) in enumerate(vocab_s):
    fwords.write(bytes("%d\t%s\t%d\n" % (i, k, v), 'UTF-8'))
fwords.close()

# create a lookup dictionary for POS tags
ftags = open(os.path.join(DATA_DIR, "alice_postags.txt"), "wb")
tags_s = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
for i, (k, v) in enumerate(tags_s):
    ftags.write(bytes("%d\t%s\t%d\n" % (i, k, v),'UTF-8'))
ftags.close()

# construct 5-grams from sentences
fgrams = open(os.path.join(DATA_DIR, "alice_5grams.txt"), "wb")
for tagged_sent in tagged_sents:
    sent_grams = []
    gram_labels = []
    # lowercase the words
    tagged_sent = [(x[0].lower(), x[1]) for x in tagged_sent]
    # replace with UNK for specific words
    tagged_sent = [(x[0] if x[0] not in vocab else "UNK", x[1])
                   for x in tagged_sent]
    # put pre- and post- padding
    tagged_sent.insert(0, ("PAD", "PAD"))
    tagged_sent.insert(0, ("PAD", "PAD"))
    tagged_sent.append(("PAD", "PAD"))
    tagged_sent.append(("PAD", "PAD"))
    for i in range(len(tagged_sent) - 4):
        sent_gram = tagged_sent[i:i+5]
        # label of middle word, and input words is 5-gram around word
        tmpstr = SPACE.join([x[0] for x in sent_gram])
        fgrams.write(bytes("%s\t%s\n" % (sent_gram[2][1], tmpstr), 'UTF-8') )
fgrams.close()