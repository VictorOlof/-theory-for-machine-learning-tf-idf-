"""
Bugs

im ligger kvar, stopword?
102 -> 'one hundred and two'
us -> u


# TODO:
# preprocessing
#   - Split
#   - clean_punctuation
#   - normalizing book
#   - remove stop words
#   - remove single characters
#   - Stemming
#   - Lemmatisation
#   - Converting numbers

stemming — need not be a dictionary word,
removes prefix and affix based on few rules

lemmatization — will be a dictionary word. reduces to a root synonym.

term frequency

tf antal gånger som ordet finns i dokumentet / antalet ord totalt



dokument 1: "hej på dig dig"
dokument 2: "hejsan på dig dig"

tf dokument 1: hej = 1/4 : på: 1/4 dig=2/4
tf dokument 2: hejsan = 1/4 : på=1/4 dig=2/4

tf antal gånger som ordet finns i dokumentet / antalet ord totalt


inverse document frequency

idf

antal dokument (N) genom antalet dokument som innehåller ordet


idf document 1: hej = 2/1 : på = 2/2 dig = 2/2
idf document 2: hejsan = 2/1 på = 2/2 dig = 2/2


tf * idf document 1:
        hej = 1/4 * 2/1) = 1/4 * 2 = 0.5, 1/4 * log(2) = 0.075
        på = 1/4 * 2/2 = 1/4 * 1 = 0.25, 1/4 * log(1) = 0


        Är summan av antalet olika ord samma sak som att beskriva hur olika de är / lika de är.


Fråga om:


bok = "hej på dig"

bok_list =  ['hej', 'på', 'dig']


bok 1 = ['det', 'är', 'fint', 'väder']
tf: hej = 0 / 4 = 0

Z tf-idf = 0.5

bok 2 = ['det', 'på', 'dig']
tf: hej = 0 / 4 = 0

Z tf-idf = 0.2
df : hej = 2/0
......

Datastruktur
använda Cosine Similarity Ranking för att representera datan som vektorer. Z tf-idf = 0.5 ska representeras (summan)

dict(ordet: (tf, df-idf))

----------->
<-----------

vektor kan ha en längd och en riktning
"""

"""
Börjar med att göra algoritmen

funktion tf, df, id-idf, preprocessing

https://machinelearningmastery.com/clean-text-machine-learning-python/
"""
