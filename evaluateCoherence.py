import os
import json
import math
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

        allCorpus += [elem['tokens'] for elem in jsonData if (elem['clusterIdSpectral'] == clusterIdSpectral and 'topicWords' in elem)]
        
    return allCorpus

def getTopicsAndLemmatizedCorpus(alluvialData, datasetType):

    allTopicIds = getAllTopicIds(alluvialData)

    jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES_CLEAN')
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

    cm = CoherenceModel(topics=allTopics, texts=lemmatizedCorpus, corpus=corpus, dictionary=dictionary, coherence='c_v')
    print('Coherence = ', format(cm.get_coherence(), '.2f'))