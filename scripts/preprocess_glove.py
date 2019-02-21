#!/usr/bin/env python3


import gensim


from models import GLOVE, GLOVE_POLEVAL_GENSIM


word_vectors = gensim.models.KeyedVectors.load_word2vec_format(GLOVE, binary=False)
word_vectors.save(GLOVE_POLEVAL_GENSIM)
print("Saved", GLOVE_POLEVAL_GENSIM)
