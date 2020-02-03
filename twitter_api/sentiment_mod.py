import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
import sys
#reload(sys)
from unidecode import unidecode
#text = unidecode(text)

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_neg = open('../python_files/negative.txt', encoding='utf-8', errors='ignore').read()
short_pos = open('../positive.txt', encoding='utf-8', errors='ignore').read()



documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features



featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)


classifier_f = open("../python_files/naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()



save_classifier = open("../naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()



voted_classifier = VoteClassifier(classifier)
                                  
#print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)
    #return(feats)
    
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
