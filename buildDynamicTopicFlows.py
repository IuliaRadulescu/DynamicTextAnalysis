import json
from os import walk
import numpy as np
from scipy.spatial import distance
import argparse

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
    @returns: a dictionary, mapping each clusterId with its centroid
    '''
    def getDiscussionClustersForSnapshot(collectionName, timeStep):

        jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES_CLEAN')
        allComments = jsonFilesDriver.readJson(collectionName)

        print('Finished reading comments for time step!', collectionName)

        discussionClustersDict = {}

        for comment in allComments:
            discussionClusterKey = str(comment['clusterIdSpectral']) + '_' + str(timeStep)
            
            if discussionClusterKey in discussionClustersDict:
                continue
            
            discussionClustersDict[discussionClusterKey] = comment['centroid']

        return discussionClustersDict

    '''
    frontsEvents = {1: {}, 2: []}
    '''
    def updateFronts(fronts, frontEvents):

        # remove the fronts that were matched with a cluster - they must be replaced by the matched cluster
        keysOfFrontsToRemove = frontEvents[1].keys()

        for keyToRemove in keysOfFrontsToRemove:
            fronts.pop(keyToRemove, None)

        # add the replacements
        for itemKey in frontEvents[1]:
            for item in frontEvents[1][itemKey]:
                if item[0] not in fronts:
                    fronts[item[0]] = item[1]
        
        # add the new created fronts
        for item in frontEvents[2]:
            fronts[item[0]] = item[1]

        return fronts

    allSnapshots = getAllCollections()

    '''
    discussionClustersMatches[clusterId_0_1] = [clusterId_1_0, clusterId_1_1, ...] 
    '''
    discussionClustersMatches = {}
    fronts = []

    snapshotDiscussionClusters0 = getDiscussionClustersForSnapshot(allSnapshots[0], 0)

    # the initial communities are the initial fronts
    # discussionClustersMatches is a mapping between clusterId_timeStep and empty list []
    discussionClustersMatches = dict(zip(snapshotDiscussionClusters0.keys(), [[] for i in range(len(snapshotDiscussionClusters0))]))

    # the fronts are expressed by the discussion clusters dictionary of the initial snapshot
    fronts = snapshotDiscussionClusters0

    for timeStep in range(1, len(allSnapshots)):

        snapshotClusters = getDiscussionClustersForSnapshot(allSnapshots[timeStep], timeStep)

        '''
        frontsEvents[frontEvent][frontId] = [front1, front2, ...]
        1 = front x was replaced by fronts list
        2 = a new front must be added
        '''
        frontEvents = {1: {}, 2: []}

        # map communities from dynamicCommunities list (t-1) to the ones in snapshot (t)
        for clusterKeySnapshot in snapshotClusters:

            clusterCentroidSnapshot = snapshotClusters[clusterKeySnapshot]
            
            bestFrontKeys = []

            for frontKey in fronts:

                centroidFront = fronts[frontKey]

                centroidSimilarity = 1 - distance.euclidean(clusterCentroidSnapshot, centroidFront)
                
                if (centroidSimilarity > optimalSim):
                    bestFrontKeys.append(frontKey)
            
            # print('BEST FRONTS', bestFrontKeys)
            if (len(bestFrontKeys) > 0):
                for bestFrontKey in bestFrontKeys:
                    # front transformation event
                    if (bestFrontKey not in frontEvents[1]):
                        frontEvents[1][bestFrontKey] = []
                    frontEvents[1][bestFrontKey].append((clusterKeySnapshot, clusterCentroidSnapshot))

                    if (bestFrontKey not in discussionClustersMatches):
                        discussionClustersMatches[bestFrontKey] = []
                    discussionClustersMatches[bestFrontKey].append(clusterKeySnapshot)
                        
            else:
                # front addition event
                frontEvents[2].append((clusterKeySnapshot, clusterCentroidSnapshot))
        
        # compute the new fronts
        # compute the new front to dynamic topic associations
        fronts = updateFronts(fronts, frontEvents)

        print('We have', len(fronts), 'fronts')

    with open(outputFileName, 'w') as outfile:
        json.dump(discussionClustersMatches, outfile)
        

parser = argparse.ArgumentParser()

parser.add_argument('-sim', '--sim', type=float, help='The minimum similarity to match communities') # for example, 0.7
parser.add_argument('-o', '--o', type=str, help='The json output file') # for example OUTPUT_TOPIC_EVOLUTION_70.json

args = parser.parse_args()

optimalSim = args.sim
outputFileName = args.o

doComputation(optimalSim, outputFileName)