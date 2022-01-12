import numpy as np
import alluvialDataRetriever
import plotAlluvial
from collections import deque

def filterAlluvialData(alluvialData, communitiesToMonitor):

    print('STARTED FILTERING ===')

    filteredAlluvial = {}
    alreadyParsedGlobal = []

    for community in communitiesToMonitor:

        if (community not in alluvialData) or (community in alreadyParsedGlobal):
            continue

        alreadyParsedGlobal.append(community)

        communitiesDeduplicated = list(set(alluvialData[community]))

        if community not in filteredAlluvial:
            filteredAlluvial[community] = communitiesDeduplicated

        communitiesQueue = deque()
        communitiesQueue.extend(communitiesDeduplicated)

        alreadyParsed = []

        while len(communitiesQueue) > 0:

            communityInQueue = communitiesQueue.popleft()

            if (communityInQueue not in alluvialData) or (communityInQueue in alreadyParsedGlobal):
                continue

            alreadyParsedGlobal.append(communityInQueue)
            
            # filter duplicates
            communitiesToAddSet = set(alluvialData[communityInQueue])

            if communityInQueue not in filteredAlluvial:
                filteredAlluvial[communityInQueue] = list(communitiesToAddSet)

            # check to see if any of the communities to add were previosly in the queue
            setDifferenceAddVsAlreadyParsed = communitiesToAddSet - set(alreadyParsed)
            listDifferenceAddVsAlredyParsed = list(setDifferenceAddVsAlreadyParsed)

            if len(listDifferenceAddVsAlredyParsed) == 0:
                continue

            alreadyParsed.extend(listDifferenceAddVsAlredyParsed)
            alreadyParsed = list(set(alreadyParsed))

            alreadyParsedGlobal.extend(listDifferenceAddVsAlredyParsed)
            alreadyParsedGlobal = list(set(alreadyParsedGlobal))
            
            communitiesQueue.extend(listDifferenceAddVsAlredyParsed)
            communitiesQueue = deque(set(communitiesQueue))

    return filteredAlluvial

def determineDynamicCommunitiesDFS(alluvialData):

    # determine leafs

    nonLeaves = list(set(alluvialData.keys()))

    alreadyParsedGlobal = set()

    stack = []
    dynamicCommunities = []

    for communityId in alluvialData:

        if (communityId not in alluvialData) or (communityId in alreadyParsedGlobal):
            continue

        stack.append((communityId, [communityId]))

        while len(stack) > 0:

            (communityId, path) = stack.pop()

            if (communityId not in alreadyParsedGlobal) and (communityId not in nonLeaves):
                dynamicCommunities.append(path)

            if (communityId not in alluvialData) or (communityId in alreadyParsedGlobal):
                continue
            
            alreadyParsedGlobal.add(communityId)
            adjacentCommunities = alluvialData[communityId]

            for adjCommunity in adjacentCommunities:
                stack.append((adjCommunity, path + [adjCommunity]))

    return dynamicCommunities

def filterAlluvialDataDFS(alluvialData, communitiesToMonitor, maxWidth = None):

    filteredAlluvial = {}

    for communityId in communitiesToMonitor:
        if (communityId in alluvialData):

            allNeighs = alluvialData[communityId]

            if maxWidth == None:
                filteredAlluvial[communityId] = allNeighs
            else:
                justMaxWidth = allNeighs[0:maxWidth]
                mandatory = list(set(communitiesToMonitor) & set(allNeighs))
                filteredAlluvial[communityId] = list(set(justMaxWidth) - set(mandatory)) + mandatory

    return filteredAlluvial    

def computeStats(fedoraFile):

    print('GENERATING STATS FOR HYBRID')

    alluvialData = alluvialDataRetriever.getAlluvialData(fedoraFile)

    print('--> width')
    communityWidths = [len(list(set(alluvialData[key]))) for key in alluvialData]
    minCommunityWidth = min(communityWidths)
    maxCommunityWidth = max(communityWidths)
    meanCommunityWidth = np.mean(communityWidths)
    print('Min width', minCommunityWidth, 'Max width', maxCommunityWidth, 'Mean width', meanCommunityWidth)

    print('--> depth')
    lenDynamicCommunities = [len(dynamicCommunity) for dynamicCommunity in determineDynamicCommunitiesDFS(alluvialData)]
    minCommunityDepth = min(lenDynamicCommunities)
    maxCommunityDepth = max(lenDynamicCommunities)
    meanCommunityDepth = np.mean(lenDynamicCommunities)
    print('Min depth', minCommunityDepth, 'Max depth', maxCommunityDepth, 'Mean depth', meanCommunityDepth)

def generateDynamicAndPlot(fedoraFile, datasetType = 'simple'):

    alluvialData = alluvialDataRetriever.getAlluvialData(fedoraFile)
    outputFileName = datasetType + '.json'

    print('STARTED DYNAMIC COMMUNITY GENERATION')

    dynamicCommunities = determineDynamicCommunitiesDFS(alluvialData)

    # sort dynamicCommunities by lentgh of dynamic communities
    dynamicCommunities.sort(key=len)

    longestDynamicItems = dynamicCommunities[len(dynamicCommunities) - 1]

    print('Longest dynamic has length', len(longestDynamicItems))

    filteredAlluvialData = filterAlluvialDataDFS(alluvialData, longestDynamicItems, maxWidth=100)

    print('STARTED PLOTTING IMAGE FOR', outputFileName)

    plotAlluvial.generateSankeyJson(filteredAlluvialData, outputFileName)

generateDynamicAndPlot('simpleText.json')
computeStats('simpleText.json')
