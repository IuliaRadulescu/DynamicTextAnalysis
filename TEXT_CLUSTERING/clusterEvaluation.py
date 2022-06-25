import random
import json
from os import walk
import numpy as np
import umap
from sklearn import preprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

class Clusterer:

    def __init__(self, fileName, preprocessedDataset, redditIds, documentVectors, docs2Tags):
        self.fileName = fileName
        self.preprocessedDataset = preprocessedDataset
        self.redditIds = redditIds
        self.documentVectors = documentVectors
        self.docs2Tags = docs2Tags
        self.dimReducer = umap.UMAP(n_components = 3, random_state = 42)

    def computeDoc2VecEmbeddings(self):
        return [self.documentVectors.get_vector(self.docs2Tags[self.fileName + '_' + str(documentNr)], norm=True) for documentNr in range(len(self.preprocessedDataset))]

    def computeCentroid(self, cluster):
        if not isinstance(cluster, np.ndarray):
            cluster = np.array(cluster)
        length, dim = cluster.shape
        centroid = np.array([np.sum(cluster[:, i])/length for i in range(dim)])

        # return normalized centroid
        return (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid))

    def getClusteringLabels(self, noClusters, commentEmbeddings):
        kMeansClusterer = KMeans(n_clusters=noClusters)
        kMeansClusterer.fit(preprocessing.normalize(commentEmbeddings))
        
        len_ = np.sqrt(np.square(kMeansClusterer.cluster_centers_).sum(axis=1)[:,None])
        centers = kMeansClusterer.cluster_centers_ / len_
        
        return (kMeansClusterer.labels_, centers)

    def doClustering(self):
        
        commentEmbeddings = np.array(self.computeDoc2VecEmbeddings())
        commentEmbeddings = self.dimReducer.fit_transform(commentEmbeddings)

        allCommentsLen = len(self.preprocessedDataset)

        # if just one comment, no need to perform clustering
        if (allCommentsLen == 1):
            return [{0: [0]}, {0: self.computeCentroid(self.commentEmbeddingsForFile)}]

        # worst case values
        maxSilhouette = -2
        maxNoClusters = 1
        bestLabels = [0] * allCommentsLen
        bestCenters = []

        for noClusters in range(min(2, (allCommentsLen - 1)), min(6, allCommentsLen)):
            
            (labels, centers) = self.getClusteringLabels(noClusters, commentEmbeddings)
            
            if (len(list(set(labels))) <= 1):    
                continue

            sscore = silhouette_score(X = commentEmbeddings, labels = labels, metric = 'cosine')

            if (sscore > maxSilhouette):
                maxSilhouette = sscore
                maxNoClusters = noClusters
                bestLabels = labels
                bestCenters = centers

        print('Best noClusters is', maxNoClusters, 'with score', sscore)

        clusterIds2Centroids = {}

        for clusterId in range(max(bestLabels) + 1):
            clusterIds2Centroids[clusterId] = bestCenters[clusterId]

        return bestLabels, commentEmbeddings

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

'''
preprocessedDocuments = a list of lists of tokens; example = [ ['Lilly', 'is', 'beautiful', 'cat'], ['Milly', 'is', 'wonderful' 'cat'] ]
https://radimrehurek.com/gensim/models/doc2vec.html
'''
def computeDoc2VecModel(vectorSize, windowSize, allComments):
    documents = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(allComments)]
    return Doc2Vec(documents=documents, vector_size=vectorSize, window=windowSize, epochs=50, dm=1, workers=30, hs=1, negative=0)

jsonFilesDriver = JsonFilesDriver('./UTILS/FEDORA_FILES_CLEAN')
fileNames = jsonFilesDriver.getAllJsonFileNames()

# create collections dictionaries
fileNames2PreprocessedRecords = {}
fileNames2RedditIds = {}
allComments = []

for fileName in fileNames:
    # an array of jsons
    jsonFileRecords = jsonFilesDriver.readJson(fileName)
    dataset = [x['body'] for x in jsonFileRecords]
    preprocessedDataset = [x['tokens'] for x in jsonFileRecords]

    allComments += preprocessedDataset

    fileNames2PreprocessedRecords[fileName] = preprocessedDataset
    fileNames2RedditIds[fileName] = [x['redditId'] for x in jsonFileRecords]

print('1 === Finished preprocessing')

docs2Tags = {}

documentsIterator = 0
for fileName in fileNames2PreprocessedRecords:
    for documentNr in range(len(fileNames2PreprocessedRecords[fileName])):
        docs2Tags[fileName + '_' + str(documentNr)] = str(documentsIterator)
        documentsIterator += 1

print('FINISHED DOCUMENTS ITERATOR', documentsIterator)

# load saved model
doc2vecModel = Doc2Vec.load('doc2VecTrainingFedora')

allSscores = []
allDBs = []
allCHs = []

randomSscores = []
randomAllDBs = []
randomAllCHs = []

imbalancedSscores = []
imbalancedAllDBs = []
imbalancedAllCHs = []

isolatedSscores = []
isolatedAllDBs = []
isolatedAllCHs = []

for fileName in fileNames2PreprocessedRecords:

    preprocessedDataset = fileNames2PreprocessedRecords[fileName]
    redditIds = fileNames2RedditIds[fileName]

    print('Evaluating collection', fileName, 'with', len(preprocessedDataset), 'comments')

    datasetClusterer = Clusterer(fileName, preprocessedDataset, redditIds, doc2vecModel.dv, docs2Tags)
    
    (labels, commentEmbeddings) = datasetClusterer.doClustering()

    sscore = silhouette_score(X = commentEmbeddings, labels = labels, metric = 'cosine')
    DB = davies_bouldin_score(commentEmbeddings, labels)
    CH = calinski_harabasz_score(commentEmbeddings, labels)

    allSscores.append(sscore)
    allDBs.append(DB)
    allCHs.append(CH)

    print('Silh = ', sscore)
    print('DB = ', DB)
    print('CH = ', CH)

    randomLabels = [random.randrange(0, len(labels) - 1, 1) for i in range(len(labels))]

    randomSscores.append(silhouette_score(X = commentEmbeddings, labels = randomLabels, metric = 'cosine'))
    randomAllDBs.append(davies_bouldin_score(commentEmbeddings, randomLabels))
    randomAllCHs.append(calinski_harabasz_score(commentEmbeddings, randomLabels))

    imbalancedLabels = [1]*(len(labels) - 1)
    imbalancedLabels.append(0)

    imbalancedSscores.append(silhouette_score(X = commentEmbeddings, labels = imbalancedLabels, metric = 'cosine'))
    imbalancedAllDBs.append(davies_bouldin_score(commentEmbeddings, imbalancedLabels))
    imbalancedAllCHs.append(calinski_harabasz_score(commentEmbeddings, imbalancedLabels))

    isolatedLabels = list(range(len(labels) - 1))
    isolatedLabels.append(len(labels) - 2)

    isolatedSscores.append(silhouette_score(X = commentEmbeddings, labels = isolatedLabels, metric = 'cosine'))
    isolatedAllDBs.append(davies_bouldin_score(commentEmbeddings, isolatedLabels))
    isolatedAllCHs.append(calinski_harabasz_score(commentEmbeddings, isolatedLabels))

    print('Evaluated collection', fileName)

    print('Clustering collection', fileName, 'END ==')

print('==== Real scores')
print('Silh = ', min(allSscores), sum(allSscores)/len(allSscores), max(allSscores))
print('DB = ', min(allDBs), sum(allDBs)/len(allDBs), max(allDBs))
print('CH = ', min(allCHs), sum(allCHs)/len(allCHs), max(allCHs))

print('==== Random scores')
print('Silh = ', min(randomSscores), sum(randomSscores)/len(randomSscores), max(randomSscores))
print('DB = ', min(randomAllDBs), sum(randomAllDBs)/len(randomAllDBs), max(randomAllDBs))
print('CH = ', min(randomAllCHs), sum(randomAllCHs)/len(randomAllCHs), max(randomAllCHs))

print('==== Imbalanced scores')
print('Silh = ', min(imbalancedSscores), sum(imbalancedSscores)/len(imbalancedSscores), max(imbalancedSscores))
print('DB = ', min(imbalancedAllDBs), sum(imbalancedAllDBs)/len(imbalancedAllDBs), max(imbalancedAllDBs))
print('CH = ', min(imbalancedAllCHs), sum(imbalancedAllCHs)/len(imbalancedAllCHs), max(imbalancedAllCHs))

print('==== Isolated scores')
print('Silh = ', min(isolatedSscores), sum(isolatedSscores)/len(isolatedSscores), max(isolatedSscores))
print('DB = ', min(isolatedAllDBs), sum(isolatedAllDBs)/len(isolatedAllDBs), max(isolatedAllDBs))
print('CH = ', min(isolatedAllCHs), sum(isolatedAllCHs)/len(isolatedAllCHs), max(isolatedAllCHs))