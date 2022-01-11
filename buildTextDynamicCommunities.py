import pymongo
import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse
import json

def doComputation(dbName, optimalSim, outputFileName):

    '''
    @returns: a list of sorted strings representing the database collections
    '''
    def getAllCollections(prefix):

        dbClient = pymongo.MongoClient('localhost', 27017)
        db = dbClient[dbName]

        allCollections = db.list_collection_names()

        allCollections = list(filter(lambda x: prefix in x, allCollections))

        dbClient.close()

        print('Finished reading all snapshots from mongo!')

        return sorted(allCollections)

    def getCommunitiesForSnapshot(collectionName, timeStep):

        def centeroidnp(arr):
            length, dim = arr.shape
            return np.array([np.sum(arr[:, i])/length for i in range(dim)])

        dbClient = pymongo.MongoClient('localhost', 27017)
        db = dbClient[dbName]

        allComments = list(db[collectionName].find())

        dbClient.close()

        print('Finished reading comments from mongo!', collectionName)

        comment2Attributes = {}

        for comment in allComments:
            comment2Attributes[comment['redditId']] = {
                    'clusterIdSpectral': comment['clusterIdSpectral'],
                    'centroid': comment['centroid']
                }
                
        timeStepDict = {}

        for redditId in comment2Attributes:
            dictKey = str(comment2Attributes[redditId]['clusterIdSpectral']) + '_' + str(timeStep)

            if dictKey not in timeStepDict:
                timeStepDict[dictKey] = [comment2Attributes[redditId]['centroid']]
            else:
                if (comment2Attributes[redditId]['centroid'] not in timeStepDict[dictKey]):
                    timeStepDict[dictKey].append(comment2Attributes[redditId]['centroid'])

        for dictKey in timeStepDict:
            timeStepDict[dictKey] = tuple(centeroidnp(np.array(timeStepDict[dictKey])))

        return timeStepDict

    '''
    frontsEvents = {1: {}, 2: []}
    '''
    def updateFronts(fronts, frontEvents, frontId2CommunityId):

        # remove things which should be removed if necessary

        indicesToRemove = frontEvents[1].keys()

        if (len(indicesToRemove) > 0):

            # !!! frontId2CommunityId needs to be updated; handle replacements and deletions

            # sort the indices to remove
            indicesToRemove = sorted(indicesToRemove)

            oldIdxToNewIdx = dict(zip(range(indicesToRemove[0]), range(indicesToRemove[0])))

            if (len(indicesToRemove) > 1):
                
                idIdx = 0
                step = 1

                while idIdx < (len(indicesToRemove) - 1):

                    currentIdxToRemove = indicesToRemove[idIdx]
                    nextIdxToRemove = indicesToRemove[idIdx + 1]
                    
                    oldIdxToNewIdx[currentIdxToRemove] = -1
                    oldIdxToNewIdx[nextIdxToRemove] = -1

                    k = currentIdxToRemove + 1

                    while k < nextIdxToRemove:
                        oldIdxToNewIdx[k] = k - step
                        k += 1

                    idIdx += 1
                    step += 1
            
            else:
                oldIdxToNewIdx[indicesToRemove[0]] = -1

                idIdx = 0
                step = 1

            # for the rest of the indices just decrement
            for idx in range(indicesToRemove[idIdx] + 1, len(frontId2CommunityId.keys())):
                oldIdxToNewIdx[idx] = idx - step

            for idxToRemove in indicesToRemove:
                if (oldIdxToNewIdx[idxToRemove] != -1):
                    print('WRONG!')

            newFrontId2CommunityId = {}

            for frontId in frontId2CommunityId:
                if (frontId in oldIdxToNewIdx) and (oldIdxToNewIdx[frontId] != -1):
                    newFrontId2CommunityId[oldIdxToNewIdx[frontId]] = frontId2CommunityId[frontId]

            frontId2CommunityId = newFrontId2CommunityId

            # !!! remove the fronts with specific indices; take care, this changes the fronts list indices
            fronts = [fronts[frontId] for frontId in range(len(fronts)) if frontId not in indicesToRemove]

            # add replacements
            for frontId in frontEvents[1]:
                for item in frontEvents[1][frontId]:
                    if item[1] not in fronts:
                        fronts += [item[1]]
                        frontId2CommunityId[len(fronts)-1] = item[0]
        
        for item in frontEvents[2]:
            if item[1] not in fronts:
                fronts += [item[1]]
                frontId2CommunityId[len(fronts)-1] = item[0]

        return (frontId2CommunityId, fronts)

    allSnapshots = getAllCollections('fiveHours')

    '''
    communitiesTimestepMapping[communityId_0_1] = [communityId_1_0, communityId_1_1, ...] 
    '''
    communitiesTimestepMapping = {}
    fronts = []
    '''
    maps each front with its associated community at the associated specific timestep
    '''
    frontId2CommunityId = {}
    timeStep = 0

    snapshotCommunities0 = getCommunitiesForSnapshot(allSnapshots[0], 0)

    # the initial communities are the initial fronts
    communitiesTimestepMapping = dict(zip(snapshotCommunities0.keys(), [[] for i in range(len(snapshotCommunities0))]))
    fronts = [item[1] for item in snapshotCommunities0.items()]
    frontId2CommunityId = dict(zip(range(len(fronts)), [communityId for communityId in snapshotCommunities0.keys()]))

    for timeStep in range(1, len(allSnapshots)):

        snapshotCommunities = getCommunitiesForSnapshot(allSnapshots[timeStep], timeStep)

        '''
        frontsEvents[frontEvent][frontId] = [front1, front2, ...]
        1 = front x was replaced by fronts list
        2 = a new front must be added
        '''
        frontEvents = {1: {}, 2: []}

        # map communities from dynamicCommunities list (t-1) to the ones in snapshot (t)
        for communityIdA in snapshotCommunities:

            centroidsTupleA = list(set(snapshotCommunities[communityIdA]))
            
            bestFrontIds = []

            for frontId in range(len(fronts)):

                centroidsTupleB = fronts[frontId] # (centroid_1, centroid_2, ..., centroid_n) - a front is actually a static community
                centroidSimilarity = dot(centroidsTupleA, centroidsTupleB)/(norm(centroidsTupleA)*norm(centroidsTupleB))
                
                if (centroidSimilarity > optimalSim):
                    # print('SIM IS BIGGER', avgSimilarity)
                    bestFrontIds.append(frontId)
            
            # print('BEST FRONTS', bestFrontIds)
            if (len(bestFrontIds) > 0):
                for bestFrontId in bestFrontIds:
                    # front transformation event
                    if (bestFrontId not in frontEvents[1]):
                        frontEvents[1][bestFrontId] = []
                    frontEvents[1][bestFrontId].append((communityIdA, centroidsTupleA))
                    if bestFrontId in frontId2CommunityId:
                        bestFrontCommunityId = frontId2CommunityId[bestFrontId]
                        communitiesTimestepMapping[bestFrontCommunityId].append(communityIdA)
            else:
                # front addition event
                frontEvents[2].append((communityIdA, centroidsTupleA))

        # update mappings
        for key in snapshotCommunities.keys():
            communitiesTimestepMapping[key] = []

        (frontId2CommunityId, fronts) = updateFronts(fronts, frontEvents, frontId2CommunityId)

        print('We have', len(fronts), 'fronts')

    finalMappings = {}

    for communityId in communitiesTimestepMapping:
        if (len(communitiesTimestepMapping[communityId]) > 0):
            finalMappings[communityId] = communitiesTimestepMapping[communityId]
        
    with open(outputFileName, 'w') as outfile:
        json.dump(finalMappings, outfile)

parser = argparse.ArgumentParser()

parser.add_argument('-db', '--db', type=str, help='The database to read from')
parser.add_argument('-sim', '--sim', type=float, help='The minimum similarity to match communities')
parser.add_argument('-o', '--o', type=str, help='The json output file')

args = parser.parse_args()

dbName = args.db
optimalSim = args.sim
outputFileName = args.o

doComputation(dbName, optimalSim, outputFileName)