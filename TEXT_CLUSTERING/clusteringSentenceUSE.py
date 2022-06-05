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

        return ' '.join(tokenizedLemmatized)

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

    def getSpectralClusteringLabels(self, noClusters):
        spectralClustering = SpectralClustering(n_clusters=noClusters, assign_labels='discretize', random_state=0)
        spectralClustering.fit(self.commentEmbeddingsForFile)
        return spectralClustering.labels_

    '''
    @return: clusterIds2FileJsonDataRecordIds, clusterIds2Centroids
    clusterIds2FileJsonDataRecordIds: mapping between each cluster id and its json files indexes
    clusterIds2Centroids: mapping between each cluster id and its centroids
    '''
    def doClustering(self):

        allEmbeddingsLen = len(self.commentEmbeddingsForFile)

        # if just one comment, no need to perform clustering
        if (allEmbeddingsLen == 1):
            return [{0: [0]}, {0: self.computeCentroid(self.commentEmbeddingsForFile)}]

        # worst case values
        maxSilhouette = -2
        maxNoClusters = 1
        bestLabels = [0] * allEmbeddingsLen

        for noClusters in range(min(2, (allEmbeddingsLen-1)), min(20, allEmbeddingsLen)):

            labels = self.getSpectralClusteringLabels(noClusters)
            
            if (len(list(set(labels))) <= 1):    
                continue

            sscore = silhouette_score(self.commentEmbeddingsForFile, labels)

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

    preprocessedDataset = [TextPreprocessor.doProcessing(comment) for comment in dataset]
    
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