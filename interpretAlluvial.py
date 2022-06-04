import numpy as np
import math
import alluvialDataRetriever
import plotAlluvial
from collections import deque

'''
@returns: list of dictionaries (the list of all topic flows)
'''
def identifyDynamicTopicFlows(alluvialData):

    dynamicTopicFlows = [] # list of dynamic topic flows dictionaries
    fullyParsedDiscussionsById = [] # discussions that are fully included in one or more topic flows; alluvialData keys whose all values belong to a topic dynamic topic flow; there's no use to start a new topic flow from them

    for discussionId in alluvialData:

        if discussionId in fullyParsedDiscussionsById or alluvialData[discussionId] == []:
            continue

        dynamicTopicFlow = {}
        dynamicTopicFlow[discussionId] = alluvialData[discussionId]

        if discussionId not in fullyParsedDiscussionsById:
            fullyParsedDiscussionsById.append(discussionId)

        discussionIdsStack = []
        discussionIdsStack.extend(alluvialData[discussionId])

        while (len(discussionIdsStack) > 0):

            stackDiscussionId = discussionIdsStack.pop()

            if (stackDiscussionId not in alluvialData or stackDiscussionId in fullyParsedDiscussionsById) or alluvialData[stackDiscussionId] == []:
                continue

            dynamicTopicFlow[stackDiscussionId] = alluvialData[stackDiscussionId]
            discussionIdsStack.extend(alluvialData[stackDiscussionId])
            
            if stackDiscussionId not in fullyParsedDiscussionsById:
                fullyParsedDiscussionsById.append(stackDiscussionId)

        dynamicTopicFlows.append(dynamicTopicFlow)

    return dynamicTopicFlows    

def computeStats(fedoraFile):

    print('GENERATING STATS...')

    alluvialData = alluvialDataRetriever.getAlluvialData(fedoraFile)

    print('--> width')
    communityWidths = [len(list(set(alluvialData[key]))) for key in alluvialData]
    minCommunityWidth = min(communityWidths)
    maxCommunityWidth = max(communityWidths)
    meanCommunityWidth = np.mean(communityWidths)
    print('Min width', minCommunityWidth, 'Max width', maxCommunityWidth, 'Mean width', meanCommunityWidth)

    print('--> depth')
    lenDynamicTopicFlows = [len(dynamicCommunity) for dynamicCommunity in identifyDynamicTopicFlows(alluvialData)]
    minCommunityDepth = min(lenDynamicTopicFlows)
    maxCommunityDepth = max(lenDynamicTopicFlows)
    meanCommunityDepth = np.mean(lenDynamicTopicFlows)
    print('Min depth', minCommunityDepth, 'Max depth', maxCommunityDepth, 'Mean depth', meanCommunityDepth)

def generateDynamicAndPlot(fedoraFile, datasetType = 'simple'):

    alluvialData = alluvialDataRetriever.getAlluvialData(fedoraFile)
    outputFileName = datasetType + '.json'

    print('STARTED DYNAMIC TOPIC FLOW GENERATION...')

    dynamicTopicFlows = identifyDynamicTopicFlows(alluvialData)

    print('FOUND', len(dynamicTopicFlows), 'TOPIC FLOWS')

    # sort dynamicTopicFlows by lentgh of dynamic communities
    dynamicTopicFlows.sort(key=len)

    longestDynamicTopic = dynamicTopicFlows[len(dynamicTopicFlows) - 1]

    print('Longest dynamic has length', len(longestDynamicTopic))

    print('STARTED PLOTTING IMAGE FOR', outputFileName)
    plotAlluvial.generateSankeyJson(longestDynamicTopic, outputFileName)

generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_50.json', 'TOPIC_EVOLUTION_50')
computeStats('OUTPUT_TOPIC_EVOLUTION_50.json')

generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_70.json', 'TOPIC_EVOLUTION_70')
computeStats('OUTPUT_TOPIC_EVOLUTION_70.json')

generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_80.json', 'TOPIC_EVOLUTION_80')
computeStats('OUTPUT_TOPIC_EVOLUTION_80.json')
