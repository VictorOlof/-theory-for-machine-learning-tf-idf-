import numpy


def tf(sentence):
    bag_of_words = {}

    sentence = sentence.split()

    for word in sentence:
        if word not in bag_of_words:
            bag_of_words[word] = 1
        else:
            bag_of_words[word] = bag_of_words[word] + 1

    for key, value in bag_of_words.items():
        bag_of_words[key] = value / len(sentence)

    return bag_of_words


def df(sentence_1, sentence_2):
    df_dict = {}

    unique_words = set(sentence_1.split()).union(set(sentence_2.split()))

    for word in unique_words:
        df_dict[word] = 0
        if word in sentence_1.split():
            df_dict[word] = df_dict[word] + 1

        if word in sentence_2.split():
            df_dict[word] = df_dict[word] + 1

    return df_dict

def idf(total_sentences, df):
    idf_dict = {}

    for key, value in df.items():
        idf_dict[key] = numpy.log10(total_sentences / df[key] + 1)
        print(df[key])
    return idf_dict

def tf_idf(tf, idf):
    tf_idf_dict = {}

    for key in tf.keys():

        tf_idf_dict[key] = tf[key] * idf[key]

    return tf_idf_dict

def matching_score(sentence_1, sentence_2):

    score = 0

    for key, value in sentence_1.items():
        if key in sentence_2:
            score = score + sentence_2[key]
    return score

def main():
    sentence_1 = "the"
    sentence_2 = "the sky is fjakdls fdlsk fdslkj fdsklj fdsjkl, fdsfd, fsdfsd"
    tf_result_1 = tf(sentence_1)
    tf_result_2 = tf(sentence_2)

    df_result = df(sentence_1, sentence_2)
    idf_result = idf(2, df_result)

    tf_idf_result_1 = tf_idf(tf_result_1, idf_result)
    tf_idf_result_2 = tf_idf(tf_result_2, idf_result)

    print(tf_idf_result_1)
    print(tf_idf_result_2)

    match = matching_score(tf_idf_result_1, tf_idf_result_2)

    print(match)























if __name__ == '__main__':
    main()
