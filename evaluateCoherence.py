import os
import json
from os import walk
import numpy as np
import re
import string
import json
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

class JsonFilesDriver:

    def __init__(self, jsonFolderName):
        self.jsonFolderName = jsonFolderName

    def readJson(self, jsonFileName):
        jsonFile = open(self.jsonFolderName + '/' + jsonFileName)
        jsonData = json.load(jsonFile)
        jsonFile.close()

        return jsonData

    def getAllJsonFileNames(self):
        fileNames = next(walk(self.jsonFolderName), (None, None, []))[2]
        return sorted(fileNames)

def getAllTopicIds(alluvialData):
    dynamicTopicFlowValues = [] 
    dynamicTopicFlowKeys = []
    dynamicTopicFlowValues += [value for values in list(alluvialData.values()) for value in values]
    dynamicTopicFlowKeys = list(alluvialData.keys())

    return list(set(dynamicTopicFlowKeys + dynamicTopicFlowValues))

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

        # remove all non-alphanumeric
        return re.sub('[^a-zA-Z0-9 \']', '', textDocument)

    @staticmethod
    def stopWordRemoval(tokenizedDocument):
        finalStop = list(get_stop_words('english')) # About 900 stopwords
        nltkWords = stopwords.words('english') # About 150 stopwords
        finalStop.extend(nltkWords)
        finalStop = list(set(finalStop))

        # filter stop words and one letter words/chars except i
        return list(filter(lambda token: (token not in finalStop), tokenizedDocument))

    @staticmethod
    def doProcessing(textDocument):
        # make lower
        textDocument = textDocument.lower()
        # reddit specific preprocessing
        textDocument = TextPreprocessor.removeLinks(textDocument)
        textDocument = TextPreprocessor.removeEmojis(textDocument)
        textDocument = TextPreprocessor.removeRedditReferences(textDocument)
        # remove wierd chars
        textDocument = TextPreprocessor.removePunctuation(textDocument)
        # decontract
        textDocument = contractions.fix(textDocument)
        # remove remaining '
        textDocument = re.sub('[^a-zA-Z0-9 ]', '', textDocument)

        # tokenize
        tokenized = word_tokenize(textDocument)

        # remove stop words
        tokenizedNoStop = TextPreprocessor.stopWordRemoval(tokenized)

        # lemmatize
        lemmatizer = WordNetLemmatizer()
        tokenizedLemmatized = [lemmatizer.lemmatize(token) for token in tokenizedNoStop]

        # too few words or no words, allow stop words
        if (len(tokenizedLemmatized) < 2):
            tokenizedLemmatized = [lemmatizer.lemmatize(token) for token in tokenized]

        # still few or no words? maybe there are just links or emojis
        if (len(tokenizedLemmatized) < 2):
            return []

        # filter empty
        tokenizedLemmatized = list(filter(lambda x: x != ' ', tokenizedLemmatized))

        return tokenizedLemmatized

def getTopicsForIds(topicIds, allCollectionsSorted, jsonFilesDriver):

    topics = []

    for topicId in topicIds:
        labelParts = topicId.split('_')
        clusterIdSpectral = int(labelParts[0])
        timeStep = int(labelParts[1])

        collectionName = allCollectionsSorted[timeStep]

        jsonData = jsonFilesDriver.readJson(collectionName)

        # if a cluster from one time interval contains more elements, skip them - all the elements of a cluster share the same topics
        alreadyParsedClusterIds = []

        for elem in jsonData:
            if (elem['clusterIdSpectral'] in alreadyParsedClusterIds or 'topicWords' not in elem):
                continue

            if (elem['clusterIdSpectral'] != clusterIdSpectral):
                continue

            alreadyParsedClusterIds.append(elem['clusterIdSpectral'])

            topics.append(elem['topicWords'])
          

    wholeDtfTopics = []

    for topic in topics:
        wholeDtfTopics.append(topic[0].split(' '))

    return wholeDtfTopics

def getLemmatizedCorpus(topicIds, allCollectionsSorted, jsonFilesDriver):

    allCorpus = []
    dictionary = []

    for topicId in topicIds:
        labelParts = topicId.split('_')
        clusterIdSpectral = int(labelParts[0])
        timeStep = int(labelParts[1])

        collectionName = allCollectionsSorted[timeStep]
        jsonData = jsonFilesDriver.readJson(collectionName)

        allCorpus += [TextPreprocessor.doProcessing(elem['body']) for elem in jsonData if (elem['clusterIdSpectral'] == clusterIdSpectral and 'topicWords' in elem)]
        
    return allCorpus

def getTopicsAndLemmatizedCorpus(alluvialData, datasetType):

    allTopicIds = getAllTopicIds(alluvialData)

    jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES')
    allCollectionsSorted = jsonFilesDriver.getAllJsonFileNames()

    allTopics = getTopicsForIds(allTopicIds, allCollectionsSorted, jsonFilesDriver)

    # filter noisy values (less that 4 words long)
    allTopics = list(filter(lambda x: len(x) == 4, allTopics))

    if (len(allTopics) == 0):
        print('No topics found')
        return

    lemmatizedCorpus = getLemmatizedCorpus(allTopicIds, allCollectionsSorted, jsonFilesDriver)

    return (allTopics, lemmatizedCorpus)

'''
Generate the files for building the Palmetto index
'''
def generateTopicsAndCorpusFiles(alluvialData, datasetType):

    if (getTopicsAndLemmatizedCorpus(alluvialData, datasetType) == None):
        return

    corpusFolderName = 'CORPUS_' + datasetType

    if not os.path.exists(corpusFolderName):
        os.makedirs(corpusFolderName)

    allTopics, lemmatizedCorpus = getTopicsAndLemmatizedCorpus(alluvialData, datasetType)

    topicsOutput = [' '.join(words) for words in allTopics]
    f = open(corpusFolderName + '/topics', 'w')
    f.write('\n'.join(topicsOutput))
    f.close()

    lemmatizedCorpusOutput = [' '.join(words) for words in lemmatizedCorpus]
    f = open(corpusFolderName + '/lemmatizedCorpus', 'w')
    f.write('\n'.join(lemmatizedCorpusOutput))
    f.close()

'''
Compute topics coherence with gensim
'''
def computeCoherence(alluvialData, datasetType):

    if (getTopicsAndLemmatizedCorpus(alluvialData, datasetType) == None):
        return

    allTopics, lemmatizedCorpus = getTopicsAndLemmatizedCorpus(alluvialData, datasetType)

    # filter noisy values (empty word lists)
    lemmatizedCorpus = list(filter(lambda x: len(x) > 0, lemmatizedCorpus))

    dictionary = corpora.Dictionary(lemmatizedCorpus)
    corpus = [dictionary.doc2bow(comment) for comment in lemmatizedCorpus]

    cm = CoherenceModel(topics=allTopics, texts=lemmatizedCorpus, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    print('Coherence = ', format(cm.get_coherence(), '.2f'))