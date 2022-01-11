import igraph
import pymongo
from collections import defaultdict
from igraph import Graph, VertexClustering
from igraph import plot
from abc import ABC, abstractclassmethod, abstractmethod
import re

class CommunityGraphBuilder(ABC):

    def __init__(self, dataset):
        self.dataset = dataset
        self.g = Graph()

    @abstractmethod
    def getEdges(self):
        return

    @abstractmethod
    def getNodes(self):
        return

    def buildCommunityGraph(self, justNodesWithEdges = False):

        (nodesList, attributesList) = self.getNodes()

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges()
        edgesList = list(filter(lambda x: x != None, edgesList))

        # print(edgesList)

        self.g.add_vertices(nodesList)
        self.g.vs[self.attributeField] = attributesList

        self.g.add_edges(edgesList)

        # remove nodes without edges
        if (justNodesWithEdges):
            nodesToRemove = [v.index for v in self.g.vs if v.degree() == 0]
            self.g.delete_vertices(nodesToRemove)

class BuildCommentsGraphGraph(CommunityGraphBuilder):

    def __init__(self, dataset, attributeField = 'clusterIdSpectral'):
        super(BuildCommentsGraphGraph, self).__init__(dataset)
        self.attributeField = attributeField

    def getNodes(self):

        nodesList = list(map(lambda x: x['redditId'], self.dataset))
        attributesList = list(map(lambda x: x[self.attributeField], self.dataset))
        
        return (nodesList, attributesList)

    def getEdges(self):

        edgesList = []
        clusterId2RedditId = defaultdict(list)

        for x in self.dataset:
            clusterId2RedditId[x[self.attributeField]].append(x['redditId'])

        for clusterId in clusterId2RedditId:
            for redditId1 in clusterId2RedditId[clusterId]:
                for redditId2 in clusterId2RedditId[clusterId]:
                    if (redditId1 != redditId2) and (((redditId1, redditId2) not in edgesList) and ((redditId2, redditId1) not in edgesList)):
                        edgesList.append((redditId1, redditId2))

        return edgesList

    def plotGraph(self, attributeField = 'clusterIdSpectral'):

        print(self.g.vs[self.attributeField])

        clusters = VertexClustering(self.g, self.g.vs[self.attributeField])

        print('MODULARITY', clusters.modularity)
        plot(clusters)

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

'''
Get all dbs in dataset
prefix - the collections must have a specific prefix
startWithCollection - start from a specified collection, alphabetically
'''
def getAllCollections(prefix, startWithCollection = False):

    def filterCollections(c, prefix, startWithCollection):
        startWithPrefix = prefix in c

        if (startWithCollection == False):
            return startWithPrefix
        
        return startWithPrefix and (c > startWithCollection)

    allCollections = db.list_collection_names()

    prefix = 'fiveHours'
    allCollections = list(filter(lambda collection: filterCollections(collection, prefix, startWithCollection), allCollections))

    return sorted(allCollections)

def getCommentsGraph(collectionName, justNodesWithEdges = False):

    allComments = list(db[collectionName].find())

    commentsGraph = BuildCommentsGraphGraph(allComments)
    commentsGraph.buildCommunityGraph(justNodesWithEdges)

    return commentsGraph

def plotCollectionGraph(collectionName, attributeField):

    community = getCommentsGraph(collectionName, False)
    community.plotGraph(attributeField)

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionFedora

plotCollectionGraph('fiveHours_28_17_00_28_22_00', 'clusterIdSpectral')