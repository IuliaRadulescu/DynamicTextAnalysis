import igraph
import json
from os import walk
from collections import defaultdict
from igraph import Graph, VertexClustering
from igraph import plot
from abc import ABC, abstractclassmethod, abstractmethod
import re

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

        plot(clusters)

def getCommentsGraph(comments, justNodesWithEdges = False):

    commentsGraph = BuildCommentsGraphGraph(comments)
    commentsGraph.buildCommunityGraph(justNodesWithEdges)
    return commentsGraph

def plotCollectionGraph(comments, attributeField):

    community = getCommentsGraph(comments, False)
    community.plotGraph(attributeField)

jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES')
plotCollectionGraph(jsonFilesDriver.readJson('twelveHours_11_22_14_00_11_23_02_00.json'), 'clusterIdSpectral')