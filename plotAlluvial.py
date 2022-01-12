import pymongo
import plotly.graph_objects as go
import random
import os
import json

# https://plotly.com/python/sankey-diagram/

def parseAlluvialData(alluvialData):

  labels = []
  source = []
  target = []
  value = []

  for key in alluvialData:
    items = alluvialData[key]
    if key not in labels:
        labels.append(key)
    for item in items:
        if item not in labels:
            labels.append(item)

  for key in alluvialData:
    items = alluvialData[key]
    source.extend([labels.index(key)]*len(items))
    itemsIndices = list(map(lambda x: labels.index(x), items))
    target.extend(itemsIndices)
    value.extend([1]*len(items))
  
  return (labels, source, target, value)

def getTopicsLabels(labels):

  def getAllCollections(prefix='fiveHours'):

    allCollections = db.list_collection_names()
    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

  topicLabels = []

  dbClient = pymongo.MongoClient('localhost', 27017)
  db = dbClient.communityDetectionFedora

  allCollectionsSorted = getAllCollections()

  for label in labels:
    labelParts = label.split('_')
    clusterIdSpectral = int(labelParts[0])
    timeStep = int(labelParts[1])

    collectionName = allCollectionsSorted[timeStep]

    topicWordsForLabel = list(db[collectionName].find({'clusterIdSpectral': clusterIdSpectral}, {'topicWords': 1, '_id': 0}))[0]

    topicLabels.append(topicWordsForLabel['topicWords'] if 'topicWords' in topicWordsForLabel else '')
  
  return topicLabels


def generateSankeyJson(alluvialData, outputFileName):

  (labels, source, target, value) = parseAlluvialData(alluvialData)

  generateRandomColor = lambda: 'rgba(' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ', 0.8)'

  def generateRandomColorList(listLen):
    return [generateRandomColor() for _ in range(listLen)]

  color = generateRandomColorList(len(source))

  data = [dict(
            type = 'sankey',
            arrangement = 'freeform',
            orientation = 'v',
            node = dict(
                label = getTopicsLabels(labels),
                pad = 10,
                color = color
            ),
            link = dict(
                source = source,
                target = target,
                value = value
            )
          )
        ]

  with open(os.path.dirname(os.path.realpath(__file__)) + '/alluvialJsons/' + outputFileName, 'w') as outfile:
    json.dump(data, outfile)