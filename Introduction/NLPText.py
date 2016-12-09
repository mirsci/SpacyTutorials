# Import spacy and English models
import spacy
# Very important, as the models are loaded in a different location
spacy.util.set_data_path('/data/projects/python3.4/spacy')
nlp = spacy.load('en')

# Process sentences 'Hello, world. Natural Language Processing in 10 lines of code.' using spaCy
doc = nlp(u'Hello, world. Natural Language Processing in 10 lines of code.')

# Get first token of the processed document
token = doc[0]
print('#First token in the document is: ', token)

# Print sentences (one sentence per line)
print('#Display each sentence separately, by line')
for sent in doc.sents:
    print(sent)

# For each token, print corresponding part of speech tag
print('#Display the POS tag for each token')
for token in doc:
    print('{} - {}'.format(token, token.pos_))

# Write a function that walks up the syntactic tree of the given token and collects all
# tokens to the root token (including root token).
def tokens_to_root(token):
    """
    Walk up the syntactic tree, collecting tokens to the root of the given `token`.
    :param token: Spacy token
    :return: list of Spacy tokens
    """
    tokens_to_r = []
    while token.head is not token:
        tokens_to_r.append(token)
        token = token.head
        tokens_to_r.append(token)

    return tokens_to_r

# For every token in document, print it's tokens to the root
print('#Display all the tokens to the root')
for token in doc:
    print('{} --> {}'.format(token, tokens_to_root(token)))

# Print dependency labels of the tokens
print('#Display dependency labels of the tokens')
for token in doc:
    print('-> '.join(['{}-{}'.format(dependent_token, dependent_token.dep_) for dependent_token in tokens_to_root(token)]))

# Print all named entities with named entity types
doc_2 = nlp(u"I went to Paris where I met my old friend Jack from uni.")
print('#Display all named entities ~ NER')
for ent in doc_2.ents:
    print('{} - {}'.format(ent, ent.label_))

# Print noun chunks for doc_2
print('#Display all noun chunks')
print([chunk for chunk in doc_2.noun_chunks])

# For every token in doc_2, print log-probability of the word, estimated from counts from a large corpus
print('#Display log-probabilities for each token, estimated from counts over a large corpus')
for token in doc_2:
    print(token, ',', token.prob)

# For a given document, calculate similarity between 'apples' and 'oranges' and 'boots' and 'hippos'
print('#Display word similarities in a document')
doc = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")
apples = doc[0]
oranges = doc[2]
boots = doc[6]
hippos = doc[8]
print('Apples similarity with oranges', apples.similarity(oranges))
print('Boots similarity with hippos', boots.similarity(hippos))

print()
# Print similarity between sentence and word 'fruit'
apples_sent, boots_sent = doc.sents
fruit = doc.vocab[u'fruit']
print('Apples similarity with Fruit', apples_sent.similarity(fruit))
print('Boots similarity with Fruit', boots_sent.similarity(fruit))