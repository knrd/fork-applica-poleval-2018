#!/usr/bin/env python3

"""
Trains per-group sequence tagging models, that is LSTM-CRFs with one bidirectional
LSTM layer and 512 hidden states on 300-dimensional GloVe embeddings,
as well as embeddings from forward and backward LMs with 2048 hidden states.
"""

from typing import List
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TokenEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import TaggedCorpus

from models import GLOVE_POLEVAL_GENSIM
from ne_groups import GROUPS
from corpora import read_group

embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings(GLOVE_POLEVAL_GENSIM),
    FlairEmbeddings('polish-forward'),  # FORWARD_LM
    FlairEmbeddings('polish-backward')  # BACKWARD_LM
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

for i, entities in enumerate(GROUPS):
    corpus: TaggedCorpus = read_group(entities)
    tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
    tagger: SequenceTagger = SequenceTagger(hidden_size=512,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=True)
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    file_name = '-'.join(entities)
    file_path = f'data/models/{file_name}'

    print(f"Training for {file_path} ({i}/{len(GROUPS)})")
    print("Tag dictionary:", tag_dictionary.idx2item)

    trainer.train(file_path,
                  learning_rate=0.05,
                  mini_batch_size=124,
                  max_epochs=40,
                  save_final_model=True,
                  test_mode=True)
