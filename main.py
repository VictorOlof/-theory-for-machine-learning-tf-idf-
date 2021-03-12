import string
from collections import Counter

import nltk
import numpy as np

from book_scraping import BookScraper


def print_book(book):
    return print(book)


class Preprocessing:
    def split(self, book):
        return book.split()

    def clean_punctuation(self, book):
        table = str.maketrans('', '', string.punctuation)
        stripped_book = [w.translate(table) for w in book]
        return stripped_book

    def normalizing_book(self, book):
        return [word.lower() for word in book]

    def remove_stop_words(self, book):
        from nltk.corpus import stopwords  # list if stopwords, 'i', 'me', 'my' etc.
        nltk.download('stopwords')

        stop_words = stopwords.words('english')
        return [word for word in book if word not in stop_words]

    def remove_single_characters(self, book):
        return [word for word in book if len(word) > 1]

    def stem_words(self, book):
        """Ordstam - Example “fishing,” “fished,” “fisher” --> fish | "Rockkks --> rockkk """
        from nltk.stem.porter import PorterStemmer

        porter = PorterStemmer()
        return [porter.stem(word) for word in book]

    def lemmatization_words(self, book):
        """Dictionary word reduced to a root synonym"""
        from nltk.stem import WordNetLemmatizer
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in book]

    def numbers_to_word(self, book):
        """ 102 --> 'one hundred and two' """
        import num2words

        new_book = []
        for word in book:
            if word.isnumeric():
                new_book.append(num2words.num2words(word))
            else:
                new_book.append(word)
        return new_book

    def unique_words(self, words):
        return set(words)


def calculate_tf(words, corpus):
    result = {}

    for word in words:
        result[word] = []

    for word in words:
        for i, document in enumerate(corpus):
            if word in set(document):
                tf = document.count(word)/len(document)
                result[word].append([i, tf])
            else:
                result[word].append([i, 0])
    return result


def calculate_df(words, corpus):
    df = {}
    for word in words:
        df[word] = 0

    for word in words:
        for book in corpus:
            if word in book:
                df[word] = df[word] + 1
    return df

def calculate_idf(df, corpus):
    idf = {}

    N = len(corpus)

    for key, df in df.items():
        if df == 0:
            idf[key] = 0
        else:
            idf[key] = np.log(N / (df + 1))

    return idf


def calculate_df_idf(tf, idf):
    tf_idf_dict = {}

    for word in tf.keys():
        tf_idf_dict[word] = []

    for word, pairs in tf.items():
        for pair in pairs:
            index, tf = pair
            tf_idf = tf * idf[word]
            tf_idf_dict[word].append([index, tf_idf])
    return tf_idf_dict

def main():
    preprocessing = Preprocessing()
    book_scraper = BookScraper()

    link_1 = "https://www.gutenberg.org/files/64651/64651-h/64651-h.htm"
    link_2 = "https://www.gutenberg.org/files/64789/64789-h/64789-h.htm"

    body_book_1 = book_scraper.get_story(link_1)
    body_book_2 = book_scraper.get_story(link_2)


    # words = preprocessing.split(body_book_1)
    words = body_book_1
    words = preprocessing.clean_punctuation(words)
    words = preprocessing.normalizing_book(words)
    words = preprocessing.remove_stop_words(words)
    words = preprocessing.remove_single_characters(words)
    words = preprocessing.numbers_to_word(words)
    # words = preprocessing.lemmatization_words(words)
    # words = preprocessing.stem_words(words)
    words = preprocessing.unique_words(words)

    book_2 = body_book_2

    # book_2 = preprocessing.split(body_book_2)
    book_2 = preprocessing.clean_punctuation(book_2)
    book_2 = preprocessing.normalizing_book(book_2)
    book_2 = preprocessing.remove_stop_words(book_2)
    book_2 = preprocessing.numbers_to_word(book_2)
    # book_2 = preprocessing.lemmatization_words(book_2)
    # book_2 = preprocessing.stem_words(book_2)

    # tf = {'hej': [('book_2_id', 'tf', 'idf', 'tf-idf'), ('book_3_id', 'tf','idf', 'tf-idf')]}

    corpus = [book_2]
    # words = ["hej", "tjena", "dig"]
    # corpus = [["hej", "hejsan"]]


    tf = calculate_tf(words, corpus)
    # print(f"tf: {tf}")
    df = calculate_df(words, corpus)
    # print(f"df {df}")
    idf = calculate_idf(df, corpus)
    # print(f"idf{idf}")

    df_idf = calculate_df_idf(tf, idf)
    print(f"df-idf{df_idf}")

    # with open('book1.txt', 'r') as file:
    #     book_1 = file.read()
    #
    # with open('book2.txt', 'r') as file:
    #     book_2 = file.read()







if __name__ == '__main__':
    main()
