import os
import json
from os import walk
import numpy as np
import re
import string
import spacy
from nltk.corpus import stopwords
from stop_words import get_stop_words
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

nlp = spacy.load('en_core_web_sm')

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

def getTopicsForIds(topicIds, allCollectionsSorted, jsonFilesDriver):

  topics = []

  for topicId in topicIds:
    labelParts = topicId.split('_')
    clusterIdSpectral = int(labelParts[0])
    timeStep = int(labelParts[1])

    collectionName = allCollectionsSorted[timeStep]

    jsonData = jsonFilesDriver.readJson(collectionName)

    for elem in jsonData:
      if (elem['clusterIdSpectral'] == clusterIdSpectral):
        if ('topicWords' in elem):
          topics.append(elem['topicWords'])
        else:
          topics.append(['missing'])
        break

  aggregatedTopics = []

  for topic in topics:
      aggregatedTopics.append(topic[0].split(' '))

  return aggregatedTopics

def getLemmatizedCorpus(topicIds, allCollectionsSorted, jsonFilesDriver):

    allCorpus = []

    # fetch only comments from existing clusters (in the alluvial diagram) with assigned topics
    existingClusters = [int(topicId.split('_')[0]) for topicId in topicIds]

    for topicId in topicIds:
        labelParts = topicId.split('_')
        clusterIdSpectral = int(labelParts[0])
        timeStep = int(labelParts[1])

        collectionName = allCollectionsSorted[timeStep]
        jsonData = jsonFilesDriver.readJson(collectionName)

        allCorpus += [TextPreprocessor.doProcessing(elem['body']) for elem in jsonData if (elem['clusterIdSpectral'] in existingClusters and 'topicWords' in elem)]
        
    return allCorpus

def computeCoherence(alluvialData):

    allTopicIds = getAllTopicIds(alluvialData)

    jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES')
    allCollectionsSorted = jsonFilesDriver.getAllJsonFileNames()

    allTopics = getTopicsForIds(allTopicIds, allCollectionsSorted, jsonFilesDriver)

    lemmatizedCorpus = getLemmatizedCorpus(allTopicIds, allCollectionsSorted, jsonFilesDriver)
    word2id = corpora.Dictionary(lemmatizedCorpus)
    corpusBow = [word2id.doc2bow(text) for text in lemmatizedCorpus]

    cm = CoherenceModel(topics=allTopics, texts=lemmatizedCorpus, dictionary=word2id, coherence='c_v')

    print('Coherence = ', cm.get_coherence())

    print('Coherence for each topic = ', cm.get_coherence_per_topic())