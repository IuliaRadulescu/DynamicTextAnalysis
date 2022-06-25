import random
import os
import json
from os import walk
import plotly.graph_objects as go

# https://plotly.com/python/sankey-diagram/

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

def getTopicsLabels(labels, allCollectionsSorted, jsonFilesDriver):

  topicLabels = []

  for label in labels:

    labelParts = label.split('_')
    clusterIdSpectral = int(labelParts[0])
    timeStep = int(labelParts[1])

    collectionName = allCollectionsSorted[timeStep]

    jsonData = jsonFilesDriver.readJson(collectionName)

    for elem in jsonData:
      if (elem['clusterIdSpectral'] == clusterIdSpectral):
        if ('topicWords' in elem):
          topicLabels.append(elem['topicWords'][0].split(' '))
        else:
          topicLabels.append('missing')
        break

  return topicLabels

def getPaths(key, alluvialData, alreadyParsed, paths, path):

    if (key not in alluvialData):
        paths.append([*path])
        path.remove(key)
        return

    if (key in alreadyParsed):
        return

    alreadyParsed.append(key)

    # re-enable matches for a valid key, so they will form a new path
    for match in alluvialData[key]:
        if (match in alreadyParsed):
            alreadyParsed.remove(match)

    for match in alluvialData[key]:        
        path.append(match)
        getPaths(match, alluvialData, alreadyParsed, paths, path)

    for match in alluvialData[key]:
        if match in path:
            path.remove(match)
    path.remove(key)

def getTopicPaths(alluvialData):

    paths = []
    alreadyParsed = []

    for key in alluvialData:
        getPaths(key, alluvialData, alreadyParsed, paths, [key])

    jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES_CLEAN')
    allCollectionsSorted = jsonFilesDriver.getAllJsonFileNames()

    topicPaths = []

    for path in paths:
      topicPaths.append(getTopicsLabels(path, allCollectionsSorted, jsonFilesDriver))

    return topicPaths

def generateSankeyJson(alluvialData, outputFileName):

  (labels, source, target, value) = parseAlluvialData(alluvialData)

  generateRandomColor = lambda: 'rgba(' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ', 0.8)'

  def generateRandomColorList(listLen):
    return [generateRandomColor() for _ in range(listLen)]

  color = generateRandomColorList(len(source))

  jsonFilesDriver = JsonFilesDriver('./TEXT_CLUSTERING/UTILS/FEDORA_FILES_CLEAN')
  allCollectionsSorted = jsonFilesDriver.getAllJsonFileNames()

  topicLabels = getTopicsLabels(labels, allCollectionsSorted, jsonFilesDriver)

  data = [dict(
            type = 'sankey',
            arrangement = 'freeform',
            orientation = 'v',
            node = dict(
                label = getTopicsLabels(labels, allCollectionsSorted, jsonFilesDriver),
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