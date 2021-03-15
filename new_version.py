import math

import numpy
import numpy as np
from numpy.dual import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tf(sentence, all_words):
    bag_of_words = {}

    sentence = sentence.split()

    for word in all_words:
        if word in sentence:
            if word not in bag_of_words:
                bag_of_words[word] = 1
            else:
                bag_of_words[word] = bag_of_words[word] + 1
        else:
            bag_of_words[word] = 0

    for key, value in bag_of_words.items():
        if value > 0:
            bag_of_words[key] = value / len(sentence)

    return bag_of_words


def df(sentence_1, sentence_2):
    df_dict = {}

    unique_words = set(sentence_1.split()).union(set(sentence_2.split()))

    for word in unique_words:
        df_dict[word] = 0
        if word in sentence_1.split():
            df_dict[word] = df_dict[word] + sentence_1.split().count(word)

        if word in sentence_2.split():
            df_dict[word] = df_dict[word] + sentence_2.split().count(word)

    return df_dict


def idf(total_sentences, df):
    idf_dict = {}

    for key, value in df.items():
        idf_dict[key] = 1 + math.log(total_sentences / df[key])

    return idf_dict


def tf_idf(tf, idf):
    tf_idf_dict = {}

    for key in tf.keys():
        tf_idf_dict[key] = tf[key] * idf[key]

    return tf_idf_dict

def cosinus_simularity2(self, vector_1, vector_2): # <-- DENNA GÄLLER
    dot_product = 0
    length_a = 0
    length_b = 0

    print("zip ", list(zip(vector_1, vector_2)))

    for a, b in list(zip(vector_1, vector_2)):
        dot_product = dot_product + a * b
    print(dot_product)
    for i in vector_1:
        length_a = length_a + i * i
    for i in vector_2:
        length_b = length_b + i * i
    length_a = numpy.sqrt(length_a)
    length_b = numpy.sqrt(length_b)

    cos = dot_product / (length_a * length_b)

    return cos


def matching_score(sentence_1, sentence_2):
    score = 1

    for key, value in sentence_1.items():
        if key in sentence_2:
            score = score + sentence_2[key]
    return score


def vectorize(tf_idf_result):
    vector = []

    for key, value in tf_idf_result.items():
        vector.append(value)

    return vector


def cosinus_similarity(a, b):
    return numpy.dot(a, b) / (norm(a) * norm(b))


def main():
    sentence_1 = "the sky is blue"
    sentence_2 = "the sky is blue"
    unique_words = set(sentence_1.split()).union(set(sentence_2.split()))

    tf_result_1 = tf(sentence_1, unique_words)
    tf_result_2 = tf(sentence_2, unique_words)

    df_result = df(sentence_1, sentence_2)
    idf_result = idf(2, df_result)

    tf_idf_result_1 = tf_idf(tf_result_1, idf_result)
    tf_idf_result_2 = tf_idf(tf_result_2, idf_result)
    # print(tf_idf_result_1)
    # print(tf_idf_result_2)

    match = matching_score(tf_idf_result_1, tf_idf_result_2)

    a = vectorize(tf_idf_result_1)
    b = vectorize(tf_idf_result_2)
    similarity = cosinus_similarity(a, b)

    print("tdidf ", tf_idf_result_1)
    print("a", a)
    print("b", b)
    print("match", match)
    print(similarity)
    #
    # corpus = [
    # #     'the sky is blue',
    # #     'the sky is not',
    # # ]
    # #
    # # vectorizer = TfidfVectorizer()
    # # tfidf = vectorizer.fit_transform(corpus)
    # # words = vectorizer.get_feature_names()
    # similarity_matrix = cosine_similarity(tfidf)
    # #
    # # print(tfidf)
    # # print(similarity_matrix)


if __name__ == '__main__':
    main()
