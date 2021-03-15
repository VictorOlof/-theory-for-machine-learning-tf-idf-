import math
import string
from collections import Counter

import nltk
import numpy


class PreProcessing:
    def preprocess(self, words):
        words = self.clean_punctuation(words)
        words = self.normalizing_book(words)
        words = self.remove_stop_words(words)
        words = self.remove_single_characters(words)
        words = self.numbers_to_word(words)
        words = self.lemmatization_words(words)
        words = self.stem_words(words)

        return words

    def tokenization(self, text):
        return text.split()

    def clean_punctuation(self, text):
        return [w.translate(str.maketrans('', '', string.punctuation)) for w in text]

    def normalizing_book(self, text):
        return [word.lower() for word in text]

    def remove_stop_words(self, text):
        from nltk.corpus import stopwords  # list if stopwords, 'i', 'me', 'my' etc.

        return [word for word in text if word not in stopwords.words('english')]

    def remove_single_characters(self, book):
        return [word for word in book if len(word) > 1]

    def stem_words(self, text):
        """Ordstam - Example “fishing,” “fished,” “fisher” --> fish """
        from nltk.stem.porter import PorterStemmer

        porter = PorterStemmer()
        return [porter.stem(word) for word in text]

    def lemmatization_words(self, text):
        """Dictionary word reduced to a root synonym"""
        from nltk.stem import WordNetLemmatizer
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in text]

    def numbers_to_word(self, text):
        """ 102 --> 'one hundred and two' """
        import num2words
        return [num2words.num2words(word) if word.isnumeric() else word for word in text]


def tf(text, unique_words):
    tf_dict = {}
    text = text.split()

    for word in unique_words:

        if word in text:

            if word not in tf_dict:
                tf_dict[word] = 1
            else:
                tf_dict[word] = tf_dict[word] + 1

        else:
            tf_dict[word] = 0

    return {k: v / len(text) for k, v in tf_dict.items() if v > 0}


def df(text_1, text_2):
    # df är antalet dokument som innehåller ett ord
    return dict(Counter(list(set(text_2.split())) + list(set(text_1.split()))))


def idf(n, df):
    return {k: 1 + math.log(n / df[k]) for k, v in df.items()}


def tf_idf(tf, idf):
    # räknar ut term frequency * invert document frequency
    return {k: tf[k] * idf[k] for k, v in tf.items()}


def cosinus_similarity(vector_1, vector_2):  # <-- DENNA GÄLLER
    dot_product = sum(a * b for a, b in list(zip(vector_1, vector_2)))
    length_a = numpy.sqrt(sum(x * x for x in vector_1))
    length_b = numpy.sqrt(sum(x * x for x in vector_2))

    return dot_product / (length_a * length_b)


def matching_score(sentence_1, sentence_2):
    return sum([sentence_2[key] for key in sentence_1.keys() if key in sentence_2])


def vectorize(dict):
    return list(dict.values())


def main():
    text_1 = "the sky is blue"
    text_2 = "the sky is blue"
    unique_words = set(text_1.split()).union(set(text_2.split()))

    tf_result_1 = tf(text_1, unique_words)
    tf_result_2 = tf(text_2, unique_words)

    df_result = df(text_1, text_2)
    idf_result = idf(2, df_result)

    tf_idf_result_1 = tf_idf(tf_result_1, idf_result)
    tf_idf_result_2 = tf_idf(tf_result_2, idf_result)
    match = matching_score(tf_idf_result_1, tf_idf_result_2)

    vector_1 = vectorize(tf_idf_result_1)
    vector_2 = vectorize(tf_idf_result_2)
    similarity = cosinus_similarity(vector_1, vector_2)

    print("tdidf ", tf_idf_result_1)
    print("a", vector_1)
    print("b", vector_2)
    print("match", match)
    print(similarity)


if __name__ == '__main__':
    main()
