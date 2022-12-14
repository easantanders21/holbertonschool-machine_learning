#!/usr/bin/env python3
"""
module 0-bag_of_words contains
function bag_of_words
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Function that creates a bag of words embedding matrix
        Arguments:
            sentences (list) : sentences to analyze
            vocab (list): contains the vocabulary words to use for the analysis
                If None, all words within sentences should be used
        Returns:
            embeddings, features
            embeddings (numpy.ndarray): shape (s, f)
                containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features (list): features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
