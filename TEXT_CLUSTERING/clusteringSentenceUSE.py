import json
from os import walk
import numpy as np
import re
import string
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import nltk
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
        tokenizedLemmas = TextPreprocessor.doLemmatization(tokenizedDocumentsNoStop)
        return ' '.join(tokenizedLemmas)
    
class Clusterer:

    def __init__(self, fileName, preprocessedDataset, fileJsonData, commentEmbeddingsForFile):
        self.fileName = fileName
        self.preprocessedDataset = preprocessedDataset
        self.fileJsonData = fileJsonData
        self.commentEmbeddingsForFile = commentEmbeddingsForFile

    def computeCentroid(self, cluster):
        if not isinstance(cluster, np.ndarray):
            cluster = np.array(cluster)
        length, dim = cluster.shape
        return np.array([np.sum(cluster[:, i])/length for i in range(dim)])

    def doClustering(self):

        allEmbeddingsLen = len(self.commentEmbeddingsForFile)

        # if just one comment, no need to perform clustering
        if (allEmbeddingsLen == 1):
            return [[0], self.commentEmbeddingsForFile[0]]

        maxSilhouette = -1
        maxNoClusters = min(3, (allEmbeddingsLen - 1))

        bestLabels = None

        for noClusters in range(min(2, (allEmbeddingsLen - 1)), min(20, allEmbeddingsLen)):
            
            spectralClustering = SpectralClustering(n_clusters=noClusters, assign_labels='discretize', random_state=0)
            spectralClustering.fit(self.commentEmbeddingsForFile)
            labels = spectralClustering.labels_

            if (len(list(set(labels))) <= 1):
                continue

            sscore = silhouette_score(self.commentEmbeddingsForFile, spectralClustering.labels_)

            if (sscore > maxSilhouette):
                maxSilhouette = sscore
                maxNoClusters = noClusters
                bestLabels = labels

        print('Best noClusters is', maxNoClusters)

        clusterIds2Embeddings = {}
        clusterIds2FileJsonDataRecordIds = {}

        for counter in range(allEmbeddingsLen):
            label = bestLabels[counter]
            embedding = self.commentEmbeddingsForFile[counter]
            
            if label not in clusterIds2Embeddings:
                clusterIds2Embeddings[label] = []
                clusterIds2FileJsonDataRecordIds[label] = []

            clusterIds2Embeddings[label].append(embedding)
            clusterIds2FileJsonDataRecordIds[label].append(counter)

        clusterIds2Centroids = {}

        for clusterId in clusterIds2Embeddings:
            clusterIds2Centroids[clusterId] = self.computeCentroid(clusterIds2Embeddings[clusterId])

        return clusterIds2FileJsonDataRecordIds, clusterIds2Centroids

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

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

jsonFilesDriver = JsonFilesDriver('./UTILS/FEDORA_FILES')
fileNames = jsonFilesDriver.getAllJsonFileNames()

print('0 === Loading USE model')

moduleUrl = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(moduleUrl)

print('0 === Loading USE model DONE')

print('1 === Preprocessing')

# create collections dictionaries
fileNames2PreprocessedRecords = {}
fileNames2Records = {}
allComments = []

for fileName in fileNames:

    jsonFileRecords = jsonFilesDriver.readJson(fileName)
    dataset = [x['body'] for x in jsonFileRecords]
    preprocessedDataset = [TextPreprocessor.doProcessing(document) for document in dataset]

    allComments += [comment for comment in preprocessedDataset]

    fileNames2PreprocessedRecords[fileName] = preprocessedDataset
    fileNames2Records[fileName] = jsonFileRecords

print('1 === Preprocessing DONE')

# embedd all comments
commentEmbeddings = model(allComments)

start = 0
end = 0

for fileName in fileNames:

    print('2 === Clustering collection', fileName)

    preprocessedDataset = fileNames2PreprocessedRecords[fileName]
    redditIds = fileNames2Records[fileName]

    end += len(preprocessedDataset)

    commentEmbeddingsForFile = commentEmbeddings[start:end]

    spectralClusterer = Clusterer(fileName, preprocessedDataset, fileNames2Records[fileName], commentEmbeddingsForFile)
    (clusterIds2FileJsonDataRecordIds, clusterIds2Centroids) = spectralClusterer.doClustering()
    
    fileJsonData = fileNames2Records[fileName]

    for clusterId in clusterIds2Centroids:
        jsonDataKeys = clusterIds2FileJsonDataRecordIds[clusterId]
        centroid = clusterIds2Centroids[clusterId]
        for jsonDataKey in jsonDataKeys:
            fileJsonData[jsonDataKey]['clusterIdSpectral'] = int(clusterId)
            fileJsonData[jsonDataKey]['centroid'] = centroid.tolist()

    jsonFilesDriver.writeJson(fileName, fileJsonData)

    start = end

    print('2 === Clustered collection', fileName)