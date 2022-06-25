import os
import json
from os import walk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np

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
            
jsonFilesDriver = JsonFilesDriver('./UTILS/FEDORA_FILES_CLEAN')
allCollections = jsonFilesDriver.getAllJsonFileNames()

for collectionName in allCollections:

    clusters2Comments = {}
    collectionRecords = jsonFilesDriver.readJson(collectionName)

    for collectionRecord in collectionRecords:
        if collectionRecord['clusterIdSpectral'] not in clusters2Comments:
            clusters2Comments[collectionRecord['clusterIdSpectral']] = [' '.join(collectionRecord['tokens'])]
        else:
            clusters2Comments[collectionRecord['clusterIdSpectral']].append(' '.join(collectionRecord['tokens']))

    for clusterId in clusters2Comments:
        
        topicExtractor = TopicExtractor(clusters2Comments[clusterId])
        topicWords = topicExtractor.getTopics(1, 5)

        jsonFilesDriver.updateByClusterId(collectionName, clusterId, topicWords)

        print('Collection Name', collectionName, 'ClusterId', clusterId, 'topics:', topicWords)