import json
import os

def getAlluvialData(fedoraFile):
    jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/FEDORA/' + fedoraFile,)
    return json.load(jsonFile)