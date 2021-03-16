class PreProcessing:
    def preprocess(self, text):
        text = self.clean_punctuation(text)
        text = self.normalizing_book(text)
        text = self.remove_stop_words(text)
        text = self.remove_single_characters(text)
        text = self.numbers_to_word(text)
        text = self.lemmatization_words(text)
        text = self.stem_words(text)
        return text

    def tokenize(self, text):
        return text.split()

    def clean_punctuation(self, text):
        import string
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
        import nltk
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in text]

    def numbers_to_word(self, text):
        """ 102 --> 'one hundred and two' """
        import num2words
        return [num2words.num2words(word) if word.isnumeric() else word for word in text]


def tf(text, unique_words):
    tf_dict = {}

    for word in unique_words:
        if word in text:
            tf_dict[word] = text.count(word)
        else:
            tf_dict[word] = 0

    print(tf_dict)

    return {k: v / len(text) for k, v in tf_dict.items() if v > 0}


def df(text_1, text_2):
    # df är antalet dokument som innehåller ett ord
    from collections import Counter
    return dict(Counter(list(set(text_2)) + list(set(text_1))))


def idf(n, df):
    import math
    return {k: math.log(n / df[k] + 1) for k, v in df.items()}


def tf_idf(tf, idf):
    # räknar ut term frequency * invert document frequency
    return {k: tf[k] * idf[k] for k, v in tf.items()}


def cosinus_similarity(vector_1, vector_2):  # <-- DENNA GÄLLER
    import numpy
    dot_product = sum(a * b for a, b in list(zip(vector_1, vector_2)))
    length_a = numpy.sqrt(sum(x * x for x in vector_1))
    length_b = numpy.sqrt(sum(x * x for x in vector_2))

    return dot_product / (length_a * length_b)


def matching_score(text_1, text_2):
    return sum([text_2[key] for key in text_1.keys() if key in text_2])


def vectorize(dict):
    return list(dict.values())


class BookScraper:
    def get_story(self, link):
        soup = self._scrape_website(link)
        story = []
        for p in soup.select("p"):
            paragraph = p.get_text().split()
            for word in paragraph:
                story.append(word)
        return story

    def _scrape_website(self, link):
        import requests
        from bs4 import BeautifulSoup

        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup

def main():
    book_scraper = BookScraper()
    pre_processing = PreProcessing()
    link_1 = "https://www.gutenberg.org/files/7353/7353-h/7353-h.htm"
    link_2 = "https://www.gutenberg.org/files/3837/3837-h/3837-h.htm"
    text_1 = book_scraper.get_story(link_1)
    text_2 = book_scraper.get_story(link_2)

    cleaned_1 = pre_processing.preprocess(text_1)
    cleaned_2 = pre_processing.preprocess(text_2)

    # cleaned_1 = ["the", "sky", "is", "not", "blue"]
    # cleaned_2 = ["the", "sky", "is", "not"]


    unique_words = set(cleaned_1).union(set(cleaned_2))

    tf_result_1 = tf(cleaned_1, unique_words)
    tf_result_2 = tf(cleaned_2, unique_words)

    df_result = df(cleaned_1, cleaned_2)
    idf_result = idf(2, df_result)

    tf_idf_result_1 = tf_idf(tf_result_1, idf_result)
    tf_idf_result_2 = tf_idf(tf_result_2, idf_result)

    match = matching_score(tf_idf_result_1, tf_idf_result_2)

    vector_1 = vectorize(tf_idf_result_1)
    vector_2 = vectorize(tf_idf_result_2)
    similarity = cosinus_similarity(vector_1, vector_2)

    print(f"Match: {match}")
    print(f"Similarity: {similarity}")


if __name__ == '__main__':
    main()