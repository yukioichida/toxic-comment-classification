# -*- coding: utf-8 -*-
import re

"""
    Class for pre process data from Jigsaw Toxic Comment dataset
"""


class PreProcessStage:

    def __init__(self, df):
        """
        :param df: DataFrame that contains input data for pre processing
        """
        self.df = df
        self.input_data = []
        self.labels = []
        self.word_2_idx = {}

    def pre_process_dataset(self):
        """
            Collects all sentences and label from dataframe and transforms into numeric representations
        """
        sentences = []
        idx = 1
        # Iterates of dataframe to collect sentences and labels
        for index, row in self.df.iterrows():
            # Normalizing and separate words of each sentence
            norm_sentence = self.norm_text(row['comment_text'])
            print(norm_sentence)
            word_sentences = re.sub("[^\w]", " ", norm_sentence).split()
            sentences.append(word_sentences)
            # Creating a word dictionary
            for word in word_sentences:
                if word not in self.word_2_idx:
                    self.word_2_idx[word] = idx
                    idx += 1
            # Getting all labels and creates a one-hot vector
            row_label = row[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
            self.labels.append(row_label)

        # Collect word indexes from prepared word dictionary
        for words_sentence in sentences:
            self.input_data.append([self.word_2_idx[w] for w in words_sentence])

    def word_index(self, word):
        return self.word_2_idx[word]

    def norm_text(self, text):
        """
        Normalizes and removes tokens contained in a sentence
        :param text: Sentence to be normalized
        :return: Sentence with normalized text
        """

        # encode to apply utf-8 and decode to remove initial 'b'
        text = str(text.encode('utf-8').decode('utf-8'))
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text
