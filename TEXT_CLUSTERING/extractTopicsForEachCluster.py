import os
import json
from os import walk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import re
import string
import spacy
from nltk.corpus import stopwords
from stop_words import get_stop_words

nlp = spacy.load('en_core_web_sm')

'''
text preprocessing pipeline - for a single unit of text corpus (a single document)
'''
class TextPreprocessor:

    @staticmethod
    def removeLinks(textDocument):
        return re.sub(r'(https?://[^\s]+)', '', textDocument)

    @staticmethod
    def removeEmojis(textDocument):
        emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
        return re.sub(emoj, '', textDocument)

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
           '`', '{', '|', '}', '~', '»', '«', '“', '”', '\n']
        pattern = re.compile("[" + re.escape("".join(specials)) + "]")
        return re.sub(pattern, '', textDocument)

    @staticmethod
    def stopWordRemoval(tokenizedDocument):
        finalStop = list(get_stop_words('english')) # About 900 stopwords
        nltkWords = stopwords.words('english') # About 150 stopwords
        finalStop.extend(nltkWords)
        finalStop.extend(['the', 'this'])
        finalStop = list(set(finalStop))

        # filter stop words and one letter words/chars except i
        return list(filter(lambda token: (token not in finalStop), tokenizedDocument))

    @staticmethod
    def doProcessing(textDocument):
        # reddit specific preprocessing
        textDocument = TextPreprocessor.removeLinks(textDocument)
        textDocument = TextPreprocessor.removeEmojis(textDocument)
        textDocument = TextPreprocessor.removeRedditReferences(textDocument)
        textDocument = TextPreprocessor.removePunctuation(textDocument)

        # tokenize and lemmatize
        processedDocument = nlp(textDocument)
        tokenizedLemmatized = [token.lemma_ for token in processedDocument]

        # generic preprocessing
        tokenizedLemmatized = TextPreprocessor.stopWordRemoval(tokenizedLemmatized)

        # too few words or no words, allow stop words
        if (len(tokenizedLemmatized) < 2):
            tokenizedLemmatized = [token.lemma_ for token in processedDocument]

        # still few or no words? maybe there are just links or emojis
        if (len(tokenizedLemmatized) < 2):
            tokenizedLemmatized = ['link', 'emoji']

        return tokenizedLemmatized

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

class JsonFilesDriver:

    def __init__(self, jsonFolderName):
        self.jsonFolderName = jsonFolderName

    def readJson(self, jsonFileName):
        jsonFile = open(self.jsonFolderName + '/' + jsonFileName)
        jsonData = json.load(jsonFile)
        jsonFile.close()
        return jsonData

    def writeJson(self, jsonFileName, data):
        jsonFile = open(self.jsonFolderName + '/' + jsonFileName, 'w')
        jsonData = json.dumps(data)
        jsonFile.write(jsonData)
        jsonFile.close()

    def getAllJsonFileNames(self):
        fileNames = next(walk(self.jsonFolderName), (None, None, []))[2]
        return sorted(fileNames)

    def updateByClusterId(self, jsonFileName, clusterId, topicWords):
        jsonData = self.readJson(jsonFileName)

        for jsonRecordId in range(len(jsonData)):
            element = jsonData[jsonRecordId]
            if (element['clusterIdSpectral'] != clusterId):
                continue
            jsonData[jsonRecordId]['topicWords'] = topicWords

        self.writeJson(jsonFileName, jsonData)
            
jsonFilesDriver = JsonFilesDriver('./UTILS/FEDORA_FILES')
allCollections = jsonFilesDriver.getAllJsonFileNames()

for collectionName in allCollections:

    clusters2Comments = {}
    collectionRecords = jsonFilesDriver.readJson(collectionName)

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

        jsonFilesDriver.updateByClusterId(collectionName, clusterId, topicWords)

        print('Collection Name', collectionName, 'ClusterId', clusterId, 'topics:', topicWords)