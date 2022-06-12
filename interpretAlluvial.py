import numpy as np
import math
import alluvialDataRetriever
import plotAlluvial
import evaluateCoherence
from collections import deque
import argparse
from datetime import date, timezone, datetime

'''
@returns reversed dictionary
reverses values with keys
'''
def reverseDictionary(dictionaryToReverse):

    reversedDictionary = {}

    for key in dictionaryToReverse:
        for value in dictionaryToReverse[key]:
            if value not in reversedDictionary:
                reversedDictionary[value] = [key]
            else:
                # if key has already been mapped to the value, don't add it twice
                if key in reversedDictionary[value]:
                    continue
                # else, append it (a key may be matched with more than one value)
                reversedDictionary[value].append(key)

    return reversedDictionary

'''
handles the case where the discussionId is not a key in alluvialData, but part of the matches list
[key] = [discussionId, some_value, some_other_value] -> then [key] must be returned - there may be several such keys

@returns: list of keys that were matched with the discussionId
'''
def searchForDiscussionIdInMatches(discussionId, alluvialDataReversed, fullyParsedDiscussionsById):

    if (discussionId in alluvialDataReversed):
        matchingKeys = list(set(alluvialDataReversed[discussionId]).difference(set(fullyParsedDiscussionsById)))

    return matchingKeys

def checkForKeyMatches(discussionIds, alluvialDataReversed, fullyParsedDiscussionsById, dynamicTopicFlow):
    # check if discussionIdMatch should be matched with other keys
    for discussionIdMatch in discussionIds:
        matchingKeys = searchForDiscussionIdInMatches(discussionIdMatch, alluvialDataReversed, fullyParsedDiscussionsById)
        for matchingKey in matchingKeys:
            if matchingKey not in dynamicTopicFlow:
                dynamicTopicFlow[matchingKey] = [discussionIdMatch]
            elif discussionIdMatch not in dynamicTopicFlow[matchingKey]:
                dynamicTopicFlow[matchingKey].append(discussionIdMatch)

'''
@returns: list of dictionaries (the list of all topic flows)
'''
def identifyDynamicTopicFlows(alluvialData):

    dynamicTopicFlows = [] # list of dynamic topic flows dictionaries
    fullyParsedDiscussionsById = [] # discussions that are fully included in one or more topic flows; alluvialData keys whose all values belong to a topic dynamic topic flow; there's no use to start a new topic flow from them
    
    alluvialDataReversed = reverseDictionary(alluvialData) # reverse alluvial data for easy matching

    for discussionId in alluvialData:

        if discussionId in fullyParsedDiscussionsById or alluvialData[discussionId] == []:
            continue

        dynamicTopicFlow = {}
        dynamicTopicFlow[discussionId] = alluvialData[discussionId]

        if discussionId not in fullyParsedDiscussionsById:
            fullyParsedDiscussionsById.append(discussionId)

        checkForKeyMatches(alluvialData[discussionId], alluvialDataReversed, fullyParsedDiscussionsById, dynamicTopicFlow)
                    
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

            checkForKeyMatches(alluvialData[stackDiscussionId], alluvialDataReversed, fullyParsedDiscussionsById, dynamicTopicFlow)

        dynamicTopicFlows.append(dynamicTopicFlow)

    return dynamicTopicFlows    

def computeStats(fedoraFile, startTimeInterval, endTimeInterval):

    print('GENERATING STATS...')

    alluvialData = alluvialDataRetriever.getAlluvialData(fedoraFile)

    # filter only desired intervals
    filteredAlluvialData = {}
    for key in alluvialData:
        timeInterval = int(key.split('_')[1])
        if timeInterval < startTimeInterval or timeInterval > endTimeInterval:
            continue
        filteredAlluvialData[key] = alluvialData[key]

    alluvialData = filteredAlluvialData

    dynamicTopicFlows = identifyDynamicTopicFlows(alluvialData)

    print('--> width')
    # the list of maximum matches in each topic flow
    # a topic flow is a dictionary, where each key is match with the list of its values
    # thus, we retrieve the number of elements in each list; then we calculate the maximum number of elements; this is the width of one topic flow
    dynamicTopicFlowWidths = [len([value for values in list(dynamicTopicFlow.values()) for value in values]) for dynamicTopicFlow in dynamicTopicFlows]

    minTopicFlowWidth = min(dynamicTopicFlowWidths)
    maxTopicFlowWidth = max(dynamicTopicFlowWidths)
    meanTopicFlowWidth = np.mean(dynamicTopicFlowWidths)
    print('Min width', minTopicFlowWidth, 'Max width', maxTopicFlowWidth, 'Mean width', meanTopicFlowWidth)

    print('--> depth')
    # the number of distinct time intervals a DTF spans on
    # we extract all the time intervals, put them in a set and the set's length is a dynamic topic flow's depth/ length
    timeIntervalsNr = []
    for dynamicTopicFlow in dynamicTopicFlows:
        dynamicTopicFlowValues = [] 
        dynamicTopicFlowKeys = []
        dynamicTopicFlowValues += [value for values in list(dynamicTopicFlow.values()) for value in values]
        dynamicTopicFlowKeys = list(dynamicTopicFlow.keys())
        allTopicIds = list(set(dynamicTopicFlowKeys + dynamicTopicFlowValues))
        allTimeIntervalIds = list(set([topicId.split('_')[1] for topicId in allTopicIds]))
        timeIntervalsNr.append(len(allTimeIntervalIds))

    minTopicFlowDepth = min(timeIntervalsNr)
    maxTopicFlowDepth = max(timeIntervalsNr)
    meanTopicFlowDepth = np.mean(timeIntervalsNr)
    print('Min depth', minTopicFlowDepth, 'Max depth', maxTopicFlowDepth, 'Mean depth', meanTopicFlowDepth)

def generateDynamicAndPlot(fedoraFile, datasetType, startTimeInterval, endTimeInterval):

    alluvialData = alluvialDataRetriever.getAlluvialData(fedoraFile)
    outputFileName = datasetType + '.json'

    # filter only desired intervals
    filteredAlluvialData = {}
    for key in alluvialData:
        timeInterval = int(key.split('_')[1])
        if timeInterval < startTimeInterval or timeInterval > endTimeInterval:
            continue
        filteredAlluvialData[key] = alluvialData[key]

    alluvialData = filteredAlluvialData

    print('STARTED DYNAMIC TOPIC FLOW GENERATION...')

    dynamicTopicFlows = identifyDynamicTopicFlows(alluvialData)

    print('FOUND', len(dynamicTopicFlows), 'TOPIC FLOWS')

    # sort dynamicTopicFlows by lentgh of dynamic communities
    dynamicTopicFlows.sort(key=len)

    longestDynamicTopic = dynamicTopicFlows[len(dynamicTopicFlows)-1]

    print('Longest dynamic has length', len(longestDynamicTopic))

    print('STARTED PLOTTING IMAGE FOR', outputFileName)
    plotAlluvial.generateSankeyJson(longestDynamicTopic, outputFileName)

    print('STARTED GENERATING TOPIC PATHS')
    topicPaths = plotAlluvial.getTopicPaths(longestDynamicTopic)

    for topicPath in topicPaths:
        
        topicsString = ' -> '.join([topic[0] for topic in topicPath])

        print(topicsString)
        print()

    print('STARTED GENERATING TOPIC COHERENCE')
    evaluateCoherence.computeCoherence(longestDynamicTopic)


parser = argparse.ArgumentParser()

parser.add_argument('-s', '--s', type=str, help='The start date as string, for example 2021-01-01') # for example 2021-01-01
parser.add_argument('-e', '--e', type=str, help='The end date as string, for example 2021-12-31') # for example 2021-12-31

args = parser.parse_args()

start = args.s
end = args.e

startTimestamp = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
endTimestamp = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
initialTimestamp = (date(2021, 1, 1).toordinal() - date(1970, 1, 1).toordinal()) * 24*60*60

intervalInSeconds = 60 * 60 * 12 # seconds (60) * minutes * hours

startTimeInterval = round((startTimestamp - initialTimestamp) / intervalInSeconds)
endTimeInterval = round((endTimestamp - initialTimestamp) / intervalInSeconds)

# generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_50.json', 'TOPIC_EVOLUTION_50', startTimeInterval, endTimeInterval)
# computeStats('OUTPUT_TOPIC_EVOLUTION_50.json', startTimeInterval, endTimeInterval)

# print()
# print()

# generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_70.json', 'TOPIC_EVOLUTION_70', startTimeInterval, endTimeInterval)
# computeStats('OUTPUT_TOPIC_EVOLUTION_70.json', startTimeInterval, endTimeInterval)

# print()
# print()

generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_80.json', 'TOPIC_EVOLUTION_80', startTimeInterval, endTimeInterval)
computeStats('OUTPUT_TOPIC_EVOLUTION_80.json', startTimeInterval, endTimeInterval)

print()
print()

generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_85.json', 'TOPIC_EVOLUTION_85', startTimeInterval, endTimeInterval)
computeStats('OUTPUT_TOPIC_EVOLUTION_85.json', startTimeInterval, endTimeInterval)

# print()
# print()

# generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_90.json', 'TOPIC_EVOLUTION_90', startTimeInterval, endTimeInterval)
# computeStats('OUTPUT_TOPIC_EVOLUTION_90.json', startTimeInterval, endTimeInterval)

# print()
# print()

# generateDynamicAndPlot('OUTPUT_TOPIC_EVOLUTION_95.json', 'TOPIC_EVOLUTION_95', startTimeInterval, endTimeInterval)
# computeStats('OUTPUT_TOPIC_EVOLUTION_95.json', startTimeInterval, endTimeInterval)
