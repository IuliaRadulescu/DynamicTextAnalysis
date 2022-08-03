import json
from os import walk
import numpy as np
from scipy.spatial import distance
import argparse
from igraph import Graph, plot

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

def doComputation(optimalSim, outputFileName):

    '''
    @returns: a list of sorted strings representing the database collections
    '''
    def getAllCollections():

        jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES_CLEAN')

        return jsonFilesDriver.getAllJsonFileNames()

    '''
    @returns: euclidean similarity (NOT distance) between centroid1 and centroid2
    '''
    def similarity(centroid1, centroid2):
        return 1 - distance.euclidean(centroid1, centroid2)

    def getEdgesList(nodeIds2Centroids1, nodeIds2Centroids2, optimalSim = None):
        edgesList = []
        edgesWeightsList = []

        nodeIds1 = list(nodeIds2Centroids1.keys())
        nodeIds1.sort()
        
        nodeIds2 = list(nodeIds2Centroids2.keys())
        nodeIds2.sort()

        for nodeId1 in nodeIds1:
            for nodeId2 in nodeIds2:
                if (nodeId1 == nodeId2):
                    continue
                if (nodeId1 > nodeId2):
                    return (edgesList, edgesWeightsList)
                clusterSim = similarity(nodeIds2Centroids1[nodeId1], nodeIds2Centroids2[nodeId2])

                if (optimalSim != None and clusterSim < optimalSim):
                    continue

                edgesList.append((nodeId1, nodeId2))
                edgesWeightsList.append(clusterSim)

        return (edgesList, edgesWeightsList)

    '''
    @input: 
        - comments: all comments at time t, marked with their clusterId and augmented with their centroids
    @returns: the snapshot represented as an igraph object
    '''
    def buildSnapshotClusterGraph(comments, timeStep):
        
        nrVertices = max([comment['clusterIdSpectral'] for comment in comments]) + 1
        nodeNames = [str(nodeId) + '_' + str(timeStep) for nodeId in range(nrVertices)]
        nodeCentroids = [comment['centroid'] for comment in comments]

        clusterIdsToCentroids = dict(zip([int(comment['clusterIdSpectral']) for comment in comments], [comment['centroid'] for comment in comments]))
        edgesList, edgesWeightsList = getEdgesList(clusterIdsToCentroids, clusterIdsToCentroids)

        g = Graph()
        g.add_vertices(nrVertices)
        g.vs['name'] = nodeNames
        g.vs['centroid'] = nodeCentroids
        g.vs['timeStep'] = [timeStep] * len(g.vs)
        g.add_edges(edgesList)
        g.es['weight'] = edgesWeightsList

        return g

    '''
    @input:
        - g1, g2: comment clustrs igraph graph
    @returns: graph g,resulting by merging g1 and g2
    '''
    def mergeGraphs(g1, g2, optimalSim):

        nrNodesG1 = len(g1.vs)
        nrNodesG2 = len(g2.vs)

        gResult = Graph.union(g1, g2)

        nodesToCentroidsG1 = {}
        for nodeId in range(0, nrNodesG1):
            nodesToCentroidsG1[nodeId] = gResult.vs[nodeId]['centroid']

        nodesToCentroidsG2 = {}
        for nodeId in range(nrNodesG1, nrNodesG2):
            nodesToCentroidsG2[nodeId] = gResult.vs[nodeId]['centroid']

        # interconnect graphs by edges
        edgesList, edgesWeightsList = getEdgesList(nodesToCentroidsG1, nodesToCentroidsG2, optimalSim)

        gResult.add_edges(edgesList)
        gResult.es['weight'] += edgesWeightsList

        return gResult

    allCollections = getAllCollections()
    snapshotClusterGraphs = []

    for timeStep in range(1, len(allCollections)):

        collectionName = allCollections[timeStep]
        jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES_CLEAN')
        collectionComments = jsonFilesDriver.readJson(collectionName)

        print('Finished reading comments for time step!', collectionName)

        snapshotClusterGraphs.append(buildSnapshotClusterGraph(collectionComments, timeStep))

    print('Finished building snapshot graphs')

    gResult = mergeGraphs(snapshotClusterGraphs[0], snapshotClusterGraphs[1], optimalSim)

    for snapshodId in range(2, len(snapshotClusterGraphs)):
        gResult = mergeGraphs(gResult, snapshotClusterGraphs[snapshodId], optimalSim)

    plot(gResult, layout=gResult.layout('kk'))

    print('Finished building DTF graph')
        
parser = argparse.ArgumentParser()

parser.add_argument('-sim', '--sim', type=float, help='The minimum similarity to match communities') # for example, 0.7
parser.add_argument('-o', '--o', type=str, help='The json output file') # for example OUTPUT_TOPIC_EVOLUTION_70.json

args = parser.parse_args()

optimalSim = args.sim
outputFileName = args.o

doComputation(optimalSim, outputFileName)