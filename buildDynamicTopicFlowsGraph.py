import json
from os import walk
import numpy as np
from scipy.spatial import distance
import argparse
import random
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

class Utils:
    def get_adjlist_with_names(graph):
        names = graph.vs["name"]
        result = {}
        for index, neighbors in enumerate(graph.get_adjlist()):
            result[names[index]] = [names[nei] for nei in neighbors]
        return result

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

    def getEdgesList(nodeIds2Centroids1, nodeIds2Centroids2, sameGraph = True, optimalSim = None, nodeNames = []):
        
        edgesList = []

        nodeIds1 = list(nodeIds2Centroids1.keys())
        nodeIds1.sort()
        
        nodeIds2 = list(nodeIds2Centroids2.keys())
        nodeIds2.sort()

        for nodeId1 in nodeIds1:
            for nodeId2 in nodeIds2:

                if (sameGraph == True and nodeId1 >= nodeId2):
                    continue

                if (sameGraph == False and len(nodeNames) > 0 and int(nodeNames[nodeId1].split('_')[1]) >= int(nodeNames[nodeId2].split('_')[1])):
                    continue
                
                clusterSim = similarity(nodeIds2Centroids1[nodeId1], nodeIds2Centroids2[nodeId2])

                if (optimalSim != None and clusterSim <= optimalSim):
                    continue

                edgesList.append((nodeId1, nodeId2))

        return edgesList

    '''
    @input: 
        - comments: all comments at time t, marked with their clusterId and augmented with their centroids
    @returns: the snapshot represented as an igraph object
    '''
    def buildSnapshotClusterGraph(comments, timeStep, optimalSim):

        nodeIdsToCentroids = dict(zip([int(comment['clusterIdSpectral']) for comment in comments], [comment['centroid'] for comment in comments]))
        nrVertices = max(list(nodeIdsToCentroids.keys())) + 1
        nodeNames = [str(nodeId) + '_' + str(timeStep) for nodeId in list(nodeIdsToCentroids.keys())]
        nodeCentroids = list(nodeIdsToCentroids.values())

        g = Graph(directed=True)
        g.add_vertices(nrVertices)
        g.vs['name'] = nodeNames
        g.vs['centroid'] = nodeCentroids

        # interconnect graphs by edges
        edgesList = getEdgesList(nodeIdsToCentroids, nodeIdsToCentroids, True, optimalSim, g.vs['name'])
        
        g.add_edges(edgesList)

        return g

    '''
    @input:
        - g1, g2: comment clustrs igraph graph
    @returns: graph g,resulting by merging g1 and g2
    '''
    def mergeGraphs(g1, g2, optimalSim):
        
        nodeIdsToCentroids = {}
        nodesToCentroidsG1 = {}
        nodesToCentroidsG2 = {}
        nodeNames = []

        nodeId = 0

        for v in g1.vs:
            nodesToCentroidsG1[nodeId] = v['centroid']
            nodeIdsToCentroids[nodeId] = v['centroid']
            nodeNames.append(v['name'])
            nodeId += 1

        for v in g2.vs:
            nodesToCentroidsG2[nodeId] = v['centroid']
            nodeIdsToCentroids[nodeId] = v['centroid']
            nodeNames.append(v['name'])
            nodeId += 1

        nrVertices = max(list(nodeIdsToCentroids.keys())) + 1
        nodeCentroids = list(nodeIdsToCentroids.values())

        gResult = Graph(directed=True)
        gResult.add_vertices(nrVertices)
        gResult.vs['name'] = nodeNames
        gResult.vs['centroid'] = nodeCentroids

        # interconnect graphs by edges
        edgesList = getEdgesList(nodesToCentroidsG1, nodesToCentroidsG2, False, optimalSim, gResult.vs['name'])
        edgesList += g1.get_edgelist() + g2.get_edgelist()
        existingEdges = gResult.get_edgelist()

        gResult.add_edges(list(set(edgesList) - set(existingEdges)))

        return gResult

    allCollections = getAllCollections()
    snapshotClusterGraphs = []

    for timeStep in range(1, len(allCollections)):

        collectionName = allCollections[timeStep]
        jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES_CLEAN')
        collectionComments = jsonFilesDriver.readJson(collectionName)

        print('Finished reading comments for time step!', collectionName)

        snapshotClusterGraphs.append(buildSnapshotClusterGraph(collectionComments, timeStep, 0.9))

    print('Finished building snapshot graphs')

    gResult = mergeGraphs(snapshotClusterGraphs[0], snapshotClusterGraphs[1], optimalSim)

    for snapshotId in range(2, len(snapshotClusterGraphs)):
        gResult = mergeGraphs(gResult, snapshotClusterGraphs[snapshotId], optimalSim)
                
    # remove nodes without edges
    gResult.vs.select(_degree=0).delete()

    plot(gResult, layout=gResult.layout('kk'))

    resultAdjList = Utils.get_adjlist_with_names(gResult)

    # remove empty values
    resultAdjList = {k: v for k, v in resultAdjList.items() if v}

    with open(outputFileName, 'w') as outfile:
        json.dump(resultAdjList, outfile)

    print('Finished building DTF graph')
        
parser = argparse.ArgumentParser()

parser.add_argument('-sim', '--sim', type=float, help='The minimum similarity to match communities') # for example, 0.7
parser.add_argument('-o', '--o', type=str, help='The json output file') # for example OUTPUT_TOPIC_EVOLUTION_70.json

args = parser.parse_args()

optimalSim = args.sim
outputFileName = args.o

doComputation(optimalSim, outputFileName)