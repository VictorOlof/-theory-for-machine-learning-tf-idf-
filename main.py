import string
from collections import Counter

import nltk
import numpy as np

from book_scraping import BookScraper


def print_book(book):
    return print(book)


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

    # def split(self, book):
    #     return book.split()
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


class Tf_Idf:
    def calc_tf(self, document):
        tf = {}
        for word in set(document):
            tf[word] = document.count(word) / len(document)
        return tf


    def calc_df(self, corpus):
        df = {}

        for document in corpus:
            for word in document:
                if word not in df.keys():
                    df[word] = 1
                else:
                    df[word] = df[word] + 1

        return df

    def calc_tf_idf(self, df_dict, corpus):
        n = len(corpus)
        tf_idf = {}

        for i in range(n):
            tokens = corpus[i]
            print(corpus[i])
            counter = len(tokens)
            tf_dict = self.calc_tf(tokens)

            for token in set(tokens):
                tf = tf_dict[token]
                df = df_dict[token]
                idf = np.log10(n + 1 / df + 1)  # eller n / (df + 1)
                # print(f"{token} idf : idf: {idf} n: {n} df: {df}")
                tf_idf[i, token] = tf * idf

        return tf_idf


def matching_score(query, tf_idf):
    tokens = {}

    for token in query:
        tokens[token] = None

    query_weights = {}

    for key in tf_idf.keys():
        if key[1] in tokens:
            if key[0] in query_weights.keys():
                query_weights[key[0]] += tf_idf[key]
            else:
                query_weights[key[0]] = tf_idf[key]

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    print(f"query weights: {query_weights}")

    return query_weights

def main():
    pp = PreProcessing()
    algorithm = Tf_Idf()
    #
    # book_scraper = BookScraper()
    #
    # link_1 = "https://www.gutenberg.org/files/64651/64651-h/64651-h.htm"
    # link_2 = "https://www.gutenberg.org/files/64789/64789-h/64789-h.htm"
    #
    # body_book_1 = book_scraper.get_story(link_1)
    # body_book_2 = book_scraper.get_story(link_2)
    #
    # book_to_compare = pp.preprocess(body_book_1)
    #
    # words = pp.unique_words(book_to_compare)
    # book_2 = pp.preprocess(body_book_2)
    # corpus = [book_2]

    corpus = [["hello"], ["the", "sky", "is", "not", "blue"], ["the", "sky", "is", "blue"]]
    df_dict = algorithm.calc_df(corpus)
    tf_idf = algorithm.calc_tf_idf(df_dict, corpus)
    matching_score(["the", "sky", "is", "not"], tf_idf)

    total_vocab = [x for x in df_dict.keys()]
    n = len(total_vocab)


if __name__ == '__main__':
    main()
