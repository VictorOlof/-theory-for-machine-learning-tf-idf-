import string


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

        stop_words = stopwords.words('english')
        return [word for word in book if word not in stop_words]

    def remove_single_characters(self, book):
        return [word for word in book if len(word) > 1]

    def stem_words(self, book):
        """Ordstam - Example “fishing,” “fished,” “fisher” --> fish | "Rockkks --> rockkk """
        from nltk.stem.porter import PorterStemmer
        print("without stem: ", book)
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


def calculate_tf(self):
    pass


def main():
    preprocessing = Preprocessing()

    file = open('book1.txt', 'rt')
    book_1 = file.read()
    file.close()
    file = open('book2.txt', 'rt')
    book_2 = file.read()
    file.close()

    book_1 = preprocessing.split(book_1)
    book_1 = preprocessing.clean_punctuation(book_1)
    book_1 = preprocessing.normalizing_book(book_1)
    book_1 = preprocessing.remove_stop_words(book_1)

    print_book(book_1)

    book_1 = preprocessing.numbers_to_word(book_1)
    book_1 = preprocessing.lemmatization_words(book_1)
    book_1 = preprocessing.stem_words(book_1)

    print_book(book_1)


if __name__ == '__main__':
    main()
