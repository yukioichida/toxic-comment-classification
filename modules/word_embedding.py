# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
from modules.log_config import LOG
import numpy


class WordEmbedding:

    def __init__(self, embedding_path, emb_dim=300):
        """
        :param embedding_path: Path of pre training word vectors
        """
        self.emb_path = embedding_path
        self.emb_dim = emb_dim

    def load_word_embedding(self):
        LOG.info("Loading word embeddings from " + self.emb_path)
        word_embeddings = KeyedVectors.load_word2vec_format(self.emb_path, binary=True)
        LOG.info("Word Embeddings loaded")
        return word_embeddings

    def load_from_vocabulary(self, vocabulary):
        """
        :param vocabulary: Word vocabulary. Expected shape
        :return: Embedding matrix (vocab_size, embedding_dimensions)
        """
        word_embeddings = self.load_word_embedding()
        word_count = len(vocabulary)
        vocab_embedding = numpy.zeros((word_count+1, self.emb_dim))
        for word, index in vocabulary.items():
            if word in word_embeddings.vocab:
                vocab_embedding[index] = word_embeddings[word]
            else:
                LOG.info("Word %s not found in embedding model" % word)
        del word_embeddings
        return vocab_embedding
