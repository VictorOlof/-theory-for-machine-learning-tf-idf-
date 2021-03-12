import string


#TODO:
# preprocessing
#   - Split
#   - clean_punctuation
#   - normalizing book
#   - remove stop words
#   - remove single characters
#   - remove single characters
#   - Stemming
#   - Lemmatisation
#   - Converting numbers









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
        pass













def calculate_tf(self):
    pass





def main():

    preprocessing = Preprocessing()

    book_1 = """It all started at Bandar Shahpur. You see, I'm a railroad construction
                man. Our job was finished, and the whole outfit was waiting at Bandar
                Shahpur, which is on the inlet Khor Musa of the Persian Gulf, for a
                boat to take us back to America.
                
                And there, out of nowhere, this Dr. Champ Chadwick showed up. He seemed
                to be starving for a little good old U.S.A. palaver, and I guess that's
                why we struck up an acquaintance.
                
                "I've been doing a little digging over in Iraq," he said offhand.
                "But things quieted down there. So now I'm bound for the desert
                and mountains to the north of here. This railroad has opened things
                up. It's difficult to get an expedition financed, you know, and
                transportation is sometimes the chief item."""


    book_2 = """I began to catch on that he was one of those guys who dig up ruins
                and things, and read a country's whole past from what they find.
                Then he went on to tell that he'd been sent out by a university in
                Pennsylvania, but that this present trip was just a sudden idea of his
                own.
                
                And as he talked I began to like Dr. Chadwick. He was a serious-faced,
                rawboned little guy--not half my size--with steady eyes, a firm chin,
                and black hair plastered down slick on his head. By and by he got
                around to mention that he was looking for a strong-backed man to take
                along with him.
                
                "I intend to strike out from Qum, the holy city," he said. "I'll try to
                get hold of a motor-truck there--and one of these desert men to drive
                it. They're rotten drivers though," he added, "and next to a dead loss
                on a trip like this." Then he sighed. "But I'm getting used to 'em."
            """

    book_1 = preprocessing.split(book_1)
    book_1 = preprocessing.clean_punctuation(book_1)
    book_1 = preprocessing.normalizing_book(book_1)

    print_book(book_1)



if __name__ == '__main__':
    main()
