def main():

    from sklearn.feature_extraction.text import TfidfVectorizer
    # list of text documents
    text = ["The dog.",
            "The dog.",
            "The ate dog."]
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(text)
    # summarize
    print(vectorizer.vocabulary_)
    print(vectorizer.idf_)
    # encode document
    vector = vectorizer.transform([text[0]])
    # summarize encoded vector
    print(vector.shape)
    print(vector.toarray())


if __name__ == '__main__':
    main()
