import json
from os import walk
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import umap

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

        return bestLabels, clusterIds2Centroids

    def updateClusters(self, labels, clusterIds2Centroids):

        clusters2RedditIds = {}

        for counter in range(len(self.redditIds)):
            
            label = labels[counter]
            redditId = self.redditIds[counter]
            
            if label not in clusters2RedditIds:
                clusters2RedditIds[label] = []

            clusters2RedditIds[label].append(redditId)

        for clusterId in clusters2RedditIds:
            jsonFilesDriver.updateWithClusterInfo(clusters2RedditIds[clusterId], int(clusterId), clusterIds2Centroids[clusterId].tolist(), self.fileName)

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

    def updateWithClusterInfo(self, redditIds, clusterId, centroid, jsonFileName):
        jsonFileRecords = self.readJson(jsonFileName)

        for jsonFileRecord in jsonFileRecords:
            if (jsonFileRecord['redditId'] not in redditIds):
                continue
            jsonFileRecord['clusterIdSpectral'] = clusterId
            jsonFileRecord['centroid'] = centroid

        self.writeJson(jsonFileName, jsonFileRecords)

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

# # compute doc2vec model
# doc2vecModel = computeDoc2VecModel(16, 3, allComments)

# print('2 === Finished doc2vec training')

# # save model to file
# doc2vecModel.save('doc2VecTrainingFedora_16')

# load saved model
doc2vecModel = Doc2Vec.load('doc2VecTrainingFedora_16')

for fileName in fileNames2PreprocessedRecords:

    preprocessedDataset = fileNames2PreprocessedRecords[fileName]
    redditIds = fileNames2RedditIds[fileName]

    print('Clustering collection', fileName, 'with', len(preprocessedDataset), 'comments')

    clusterer = Clusterer(fileName, preprocessedDataset, redditIds, doc2vecModel.dv, docs2Tags)
    
    (labels, clusterIds2Centroids) = clusterer.doClustering()

    clusterer.updateClusters(labels, clusterIds2Centroids)

    print('Clustering collection', fileName, 'END ==')