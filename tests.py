import unittest
from main import calculate_df
class id_idf(unittest.TestCase):

    def test_tf(self):
        pass

    def test_df_same(self):
        words = ["hej", "på", "dig"]
        book = ["hej", "på", "dig"]
        corpus = [book]
        result_df = calculate_df(words, corpus)
        expected_df = {'hej': 1, 'på': 1, 'dig': 1}

        self.assertEqual(result_df, expected_df)

    def test_df_not_same(self):
        words = ["det", "är", "soligt", "idag"]
        book = ["hej", "på", "dig"]
        corpus = [book]
        result_df = calculate_df(words, corpus)
        expected_df = {'det': 0, 'idag': 0, 'soligt': 0, 'är': 0}
        self.assertEqual(result_df, expected_df)

    def test_df_difference(self):
        words = ["det", "är", "soligt", "idag"]
        book = ["soligt", "idag"]
        corpus = [book]
        result_df = calculate_df(words, corpus)
        expected_df = {'det': 0, 'idag': 1, 'soligt': 1, 'är': 0}
        self.assertEqual(result_df, expected_df)









