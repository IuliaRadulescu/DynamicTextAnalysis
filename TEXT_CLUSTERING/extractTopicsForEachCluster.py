import pymongo
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from stop_words import get_stop_words

'''
text preprocessing pipeline - for a single unit of text corpus (a single document)
'''
class TextPreprocessor:

    @staticmethod
    def removeLinks(textDocument):
        return re.sub(r'(https?://[^\s]+)', '', textDocument)

    @staticmethod
    def removeRedditReferences(textDocument):
        return re.sub(r'(/r/[^\s]+)', '', textDocument)

    @staticmethod
    def removePunctuation(textDocument):
        # remove 'normal' punctuation
        textDocument = textDocument.strip(string.punctuation)

        # remove special chars
        specials = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '.',
           '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', 
           '`', '{', '|', '}', '~', '»', '«', '“', '”']
        pattern = re.compile("[" + re.escape("".join(specials)) + "]")
        return re.sub(pattern, '', textDocument)

    @staticmethod
    def stopWordRemoval(tokenizedDocument):
        finalStop = list(get_stop_words('english')) # About 900 stopwords
        nltkWords = stopwords.words('english') # About 150 stopwords
        finalStop.extend(nltkWords)
        finalStop.extend(['like', 'the', 'this'])
        finalStop = list(set(finalStop))

        # filter stop words and one letter words/chars except i
        tokenizedDocumentsNoStop = list(filter(lambda token: (token not in finalStop) and (len(token) > 1 and token != 'i'), tokenizedDocument))
        return list(filter(lambda token: len(token) > 0, tokenizedDocumentsNoStop))

    @staticmethod
    def doLemmatization(tokenizedDocument):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokenizedDocument]

    @staticmethod
    def doProcessing(textDocument):
        # reddit specific preprocessing
        textDocument = TextPreprocessor.removeLinks(textDocument)
        textDocument = TextPreprocessor.removeRedditReferences(textDocument)
        textDocument = TextPreprocessor.removePunctuation(textDocument)

        # tokenize
        tokenizedDocument = word_tokenize(textDocument.lower())

        # generic preprocessing
        tokenizedDocumentsNoStop = TextPreprocessor.stopWordRemoval(tokenizedDocument)
        return TextPreprocessor.doLemmatization(tokenizedDocumentsNoStop)

class TopicExtractor:

    def __init__(self, comments):
        self.comments = comments

    def prepareForLDA(self):

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.comments)

        return (vectorizer, X)
    
    # Helper function
    def prettyPrintTopics(self, model, count_vectorizer, n_top_words, printResults = False):
        words = count_vectorizer.get_feature_names()
        if 'the' in words:
            print('THE IN WORDS')
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            wordsInTopic = ' '.join([words[i]
                                for i in topic.argsort()[:-n_top_words:-1]])
            topics.append(wordsInTopic)
            if (printResults):
                print('\nTopic #%d:' % topic_idx)
                print('Words:', wordsInTopic)
        
        return topics
    
    def getTopics(self, noTopics, noWords):

        vectorizer, X = self.prepareForLDA()

        lda = LDA(n_components = noTopics, n_jobs = -1)
        lda.fit(X) # Print the topics found by the LDA model
        
        print("Topics found via LDA:")
        topics = self.prettyPrintTopics(lda, vectorizer, noWords, False)

        return topics

class MongoDBClient:

    __instance = None

    def __init__(self):

        if MongoDBClient.__instance != None:
            raise Exception('The MongoDBClient is a singleton')
        else:
            MongoDBClient.__instance = self

        self.dbClient = pymongo.MongoClient('localhost', 27017)

    @staticmethod
    def getInstance():
        
        if MongoDBClient.__instance == None:
            MongoDBClient()

        return MongoDBClient.__instance

def getAllCollections(prefix='fiveHours'):

    allCollections = db.list_collection_names()
    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionFedora
dbClient.close()

allCollections = getAllCollections()

for collectionName in allCollections:
    dbClient = pymongo.MongoClient('localhost', 27017)

    clusters2Comments = {}
    collectionRecords = list(db[collectionName].find())

    for collectionRecord in collectionRecords:
        if collectionRecord['clusterIdSpectral'] not in clusters2Comments:
            # remove [deleted] comments
            if (collectionRecord['body'] != '[deleted]'):
                clusters2Comments[collectionRecord['clusterIdSpectral']] = [collectionRecord['body']]
        else:
            clusters2Comments[collectionRecord['clusterIdSpectral']].append(collectionRecord['body'])

    for clusterId in clusters2Comments:
        preprocessedList = [TextPreprocessor.doProcessing(comment) for comment in clusters2Comments[clusterId]]
        preprocessed = [item for sublist in preprocessedList for item in sublist]
        topicExtractor = TopicExtractor(preprocessed)
        topicWords = topicExtractor.getTopics(1, 5)

        db = dbClient.communityDetectionFedora

        db[collectionName].update_many(
            {
            'clusterIdSpectral': clusterId
            },{
                '$set': {
                    'topicWords': topicWords
                }
            })

        print('Collection Name', collectionName, 'ClusterId', clusterId, 'topics:', topicWords)

    dbClient.close()