import json
from os import walk
import numpy as np
import emoji
import re
import string
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

'''
text preprocessing pipeline - for a single unit of text corpus (a single document)
'''
class TextPreprocessor:
    @staticmethod
    def removeLinks(textDocument):
        return re.sub(r'(https?://[^\s]+)', '', textDocument)

    @staticmethod
    def removeEmojis(textDocument):
        emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
        return re.sub(emoj, '', textDocument)

    @staticmethod
    def removeRedditReferences(textDocument):
        return re.sub(r'(/r/[^\s]+)', '', textDocument)

    @staticmethod
    def removeCodeAndNonASCII(textDocument):
        # remove code snippets
        textDocument = re.sub(r'```.+?```', '', textDocument)
        textDocument = re.sub(r'``.+?``', '', textDocument)
        textDocument = re.sub(r'`.+?`', '', textDocument)

        # remove xmls
        textDocument = re.sub(r'<.+?>', '', textDocument)
        return re.sub('[^a-zA-Z0-9 \'?!,.]', '', textDocument)

    @staticmethod
    def stopWordRemoval(tokenizedDocument):
        finalStop = list(get_stop_words('english')) # About 900 stopwords
        nltkWords = stopwords.words('english') # About 150 stopwords
        finalStop.extend(nltkWords)
        finalStop = list(set(finalStop))

        # filter stop words and one letter words/chars except i
        return list(filter(lambda token: (token not in finalStop), tokenizedDocument))

    @staticmethod
    def doCleaning(textDocument):
        # make lower
        textDocument = textDocument.lower()
        # reddit specific preprocessing
        textDocument = TextPreprocessor.removeLinks(textDocument)
        textDocument = TextPreprocessor.removeEmojis(textDocument)
        textDocument = TextPreprocessor.removeRedditReferences(textDocument)
        textDocument = TextPreprocessor.removeCodeAndNonASCII(textDocument)
        # decontract
        textDocument = contractions.fix(textDocument)
        # remove remaining '
        textDocument = re.sub('[^a-zA-Z0-9 ?!,.]', '', textDocument)

        # tokenize
        tokenized = word_tokenize(textDocument)

        # filter empty
        tokenized = list(filter(lambda x: x.strip() != '', tokenized))

        if (len(tokenized) == 0):
            return False

        return ' '.join(tokenized)

    @staticmethod
    def doProcessing(textDocument):

        # remove everything that is not letter, number, or space
        textDocument = re.sub('[^a-zA-Z0-9 ]', '', textDocument)

        # tokenize
        tokenized = word_tokenize(textDocument)

        # remove stop words
        tokenizedNoStop = TextPreprocessor.stopWordRemoval(tokenized)

        finalTokens = [lemmatizer.lemmatize(token) for token in tokenizedNoStop]

        if (len(finalTokens) < 4):
            return False

        return finalTokens

class JsonFilesDriver:

    def __init__(self, jsonFolderName):
        self.jsonFolderName = jsonFolderName

    def readJson(self, jsonFileName):
        jsonFile = open(self.jsonFolderName + '/' + jsonFileName)
        jsonData = json.load(jsonFile)
        jsonFile.close()

        return jsonData

    def writeJson(self, jsonFileName, data):
        jsonFile = open(self.jsonFolderName + '/' + jsonFileName, 'w')
        jsonData = json.dumps(data)
        jsonFile.write(jsonData)
        jsonFile.close()

    def getAllJsonFileNames(self):
        fileNames = next(walk(self.jsonFolderName), (None, None, []))[2]
        return sorted(fileNames)

jsonFilesDriverRaw = JsonFilesDriver('./UTILS/FEDORA_FILES_RAW')
fileNames = jsonFilesDriverRaw.getAllJsonFileNames()

jsonFilesDriverClean = JsonFilesDriver('./UTILS/FEDORA_FILES_CLEAN')

for fileName in fileNames:
    jsonFileRecords = jsonFilesDriverRaw.readJson(fileName)
    newJsonFileRecords = []

    for jsonFileRecord in jsonFileRecords:
        if (jsonFileRecord['body'] == '[deleted]' or jsonFileRecord['body'] == '[removed]'):
            continue

        cleanBody = TextPreprocessor.doCleaning(jsonFileRecord['body'])
        if (cleanBody == False):
            continue

        tokenizedBody = TextPreprocessor.doProcessing(cleanBody)
        if (tokenizedBody == False):
            continue

        jsonFileRecord['body'] = cleanBody
        jsonFileRecord['tokens'] = tokenizedBody

        newJsonFileRecords.append(jsonFileRecord)

    if (len(newJsonFileRecords) == 0):
        print('The file', fileName, 'was empty')
        continue
    
    jsonFilesDriverClean.writeJson(fileName, newJsonFileRecords)