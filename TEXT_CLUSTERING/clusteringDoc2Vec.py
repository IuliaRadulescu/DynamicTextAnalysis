import pymongo
from datetime import datetime
import numpy as np
import re
import string
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
        return TextPreprocessor.doLemmatization(tokenizedDocumentsNoStop)
    
class Clusterer:

    def __init__(self, collectionName, preprocessedDataset, redditIds, documentVectors, docs2Tags):
        self.collectionName = collectionName
        self.preprocessedDataset = preprocessedDataset
        self.redditIds = redditIds
        self.documentVectors = documentVectors
        self.docs2Tags = docs2Tags

    def computeDoc2VecEmbeddings(self):
        X = [self.documentVectors.get_vector(self.docs2Tags[self.collectionName + '_' + str(documentNr)], norm=True) for documentNr in range(len(self.preprocessedDataset))]
        return X

    def computeCentroid(self, cluster):
        print(cluster)
        if not isinstance(cluster, np.ndarray):
            cluster = np.array(cluster)
        length, dim = cluster.shape
        return np.array([np.sum(cluster[:, i])/length for i in range(dim)])

    def doClustering(self):
        X = np.array(self.computeDoc2VecEmbeddings())

        allCommentsLen = len(self.preprocessedDataset)

        # if just one comment, no need to perform clustering
        if (allCommentsLen == 1):
            return [[0], X]

        maxSilhouette = -1
        maxNoClusters = min(3, (allCommentsLen - 1))

        for noClusters in range(min(2, (allCommentsLen - 1)), min(20, allCommentsLen)):
            
            spectralClustering = SpectralClustering(n_clusters=noClusters, assign_labels='discretize', random_state=0)
            spectralClustering.fit(X)

            if (len(list(set(spectralClustering.labels_))) <= 1):
                continue

            sscore = silhouette_score(X, spectralClustering.labels_)

            if (sscore > maxSilhouette):
                maxSilhouette = sscore
                maxNoClusters = noClusters

            # print('Silhouette for', noClusters, 'is', sscore)

        print('Best noClusters is', maxNoClusters)

        spectralClustering = SpectralClustering(n_clusters=maxNoClusters, assign_labels='discretize', random_state=0)
        spectralClustering.fit(X)

        labels = spectralClustering.labels_

        clusterIds2Embeddings = {}

        for counter in range(len(X)):
            label = labels[counter]
            redditId = X[counter]
            
            if label not in clusterIds2Embeddings:
                clusterIds2Embeddings[label] = []

            clusterIds2Embeddings[label].append(redditId)

        clusterIds2Centroids = {}

        for clusterId in clusterIds2Embeddings:
            clusterIds2Centroids[clusterId] = self.computeCentroid(clusterIds2Embeddings[clusterId])

        return labels, clusterIds2Centroids

    def updateClusters(self, labels, labels2Centroids):

        clusters2RedditIds = {}

        for counter in range(len(self.redditIds)):
            
            label = labels[counter]
            redditId = self.redditIds[counter]
            
            if label not in clusters2RedditIds:
                clusters2RedditIds[label] = []

            clusters2RedditIds[label].append(redditId)

        for clusterId in clusters2RedditIds:
            MongoDBClient.getInstance().updateComments(clusters2RedditIds[clusterId], int(clusterId), labels2Centroids[clusterId].tolist(), self.collectionName)


class MongoDBClient:

    __instance = None

    def __init__(self):

        if MongoDBClient.__instance != None:
            raise Exception('The MongoDBClient is a singleton')
        else:
            MongoDBClient.__instance = self

    @staticmethod
    def getInstance():
        
        if MongoDBClient.__instance == None:
            MongoDBClient()

        return MongoDBClient.__instance

    def updateComments(self, redditIds, clusterId, centroid, collectionName):

        self.dbClient = pymongo.MongoClient('localhost', 27017)

        db = self.dbClient.communityDetectionFedora

        db[collectionName].update_many(
            {
            'redditId': {
                '$in': redditIds
                }
            },{
                '$set': {
                    'clusterIdSpectral': clusterId,
                    'centroid': centroid
                }
            })

        self.dbClient.close()


def getAllCollections(prefix='fiveHours'):

    allCollections = db.list_collection_names()
    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

'''
preprocessedDocuments = a list of lists of tokens; example = [ ['Lilly', 'is', 'beautiful', 'cat'], ['Milly', 'is', 'wonderful' 'cat'] ]
https://radimrehurek.com/gensim/models/doc2vec.html
'''
def computeDoc2VecModel(vectorSize, windowSize, allDocuments):
    documents = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(allDocuments)]
    return Doc2Vec(documents, vector_size=vectorSize, window=windowSize, epochs=50, dm=1, workers=20)

nltk.download('punkt')
nltk.download('stopwords')

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionFedora

allCollections = getAllCollections()

# compute doc2vec model

# create collections dictionaries
collections2Documents = {}
collections2RedditIds = {}
allDocuments = []

for collectionName in allCollections:
    allRecords = list(db[collectionName].find())
    dataset = [x['body'] for x in allRecords]
    preprocessedDataset = [TextPreprocessor.doProcessing(document) for document in dataset]

    allDocuments += preprocessedDataset

    collections2Documents[collectionName] = preprocessedDataset
    collections2RedditIds[collectionName] = [x['redditId'] for x in allRecords]

dbClient.close()

# print('1 === Finished preprocessing')

docs2Tags = {}

documentsIterator = 0
for collectionName in collections2Documents:
    for documentNr in range(len(collections2Documents[collectionName])):
        docs2Tags[collectionName + '_' + str(documentNr)] = str(documentsIterator)
        documentsIterator += 1

print('FINISHED DOCUMENTS ITERATOR', documentsIterator)

# compute doc2vec model
# 24 neurons (vector size) and 3 words window - because we have small documents
# doc2vecModel = computeDoc2VecModel(24, 3, allDocuments)

# print('2 === Finished doc2vec training')

# save model to file
# doc2vecModel.save('doc2VecTrainingFedora')

doc2vecModel = Doc2Vec.load('doc2VecTrainingFedora')

for collectionName in collections2Documents:

    preprocessedDataset = collections2Documents[collectionName]
    redditIds = collections2RedditIds[collectionName]

    print('Clustering collection', collectionName, 'with', len(preprocessedDataset), 'comments')

    spectralClusterer = Clusterer(collectionName, preprocessedDataset, redditIds, doc2vecModel.dv, docs2Tags)
    
    (labels, labels2Centroids) = spectralClusterer.doClustering()

    spectralClusterer.updateClusters(labels, labels2Centroids)

    print('Clustering collection', collectionName, 'END ==')