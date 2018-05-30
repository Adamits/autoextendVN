import torch
import numpy as np
import gensim
import pickle

import verbnet_extractor

w_rels, c_rels = verbnet_extractor.extract_relations()
words = list(w_rels.keys())
classes = list(c_rels.keys())
print("Loading embeddings from binary file...")
embeds = gensim.models.KeyedVectors.load_word2vec_format(\
                './data/embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)

print("Scoping embeddings to VerbNet lexicon...")
# Scope the set of words to those that are actually in the
# w2v Embeddings matrix. This excludes verbs with e.g. an '_'
valid_words = [w for w in words if w in embeds.vocab]
word2word_vector = {w: embeds[w] for w in valid_words}
word2id = {w: i for i, w in enumerate(valid_words)}
class2id = {c: i for i, c in enumerate(classes)}

# Transform the dict of word -> class relations to their ids
# in the embeddings dict, cleaning the list to drop words/classes
# That did not exist in the pre-trained embeddings
w_rels_ids = {word2id[w]: [class2id[c] for c in c_list]\
              for w, c_list in w_rels.items() if w in valid_words}
# Do the same, but for the class -> word relations
c_rels_ids = {class2id[c]: [word2id[w] for w in w_list\
                    if w in valid_words] for c, w_list in c_rels.items()}

# hardcode 300 for dim size
embeds = torch.FloatTensor(len(word2word_vector), 300)

for w, wv in word2word_vector.items():
    embeds[word2id[w]] = torch.from_numpy(wv)

word2id_file = open("./data/word2id", "wb")
class2id_file = open("./data/class2id", "wb")
word2class_file = open("./data/word2class", "wb")
class2word_file = open("./data/class2word", "wb")
vnwordsembedded_file = open("./data/vnwords_embedded", "wb")

pickle.dump(w_rels_ids, word2class_file)
pickle.dump(c_rels_ids, class2word_file)
pickle.dump(word2id, word2id_file)
pickle.dump(class2id, class2id_file)
pickle.dump(embeds, vnwordsembedded_file)
