import pymongo
import math
from datetime import date
from datetime import datetime
from random_object_id import generate

def createTimeIntervals(startTimestamp, endTimestamp, intervalInSeconds):

    timeIntervals = []

    current = startTimestamp

    while (current < endTimestamp):
        
        nextTimestamp = current + intervalInSeconds

        timeIntervals.append(str(current) + '_' + str(nextTimestamp))

        current = nextTimestamp

    return timeIntervals

def convertTimestampIntervalToReadableInterval(timestampInterval):

    timestamps = timestampInterval.split('_')

    datetimeObj1 = datetime.fromtimestamp(int(timestamps[0]))
    datetimeObj2 = datetime.fromtimestamp(int(timestamps[1]))

    day1Padded0 = str(datetimeObj1.day) if int(datetimeObj1.day) > 9 else str(datetimeObj1.day).zfill(2)
    day2Padded0 = str(datetimeObj2.day) if int(datetimeObj2.day) > 9 else str(datetimeObj2.day).zfill(2)

    humanReadableInterval  = 'fiveHours_' \
        + day1Padded0 + '_' + str(datetimeObj1.strftime('%H')) + '_' + str(datetimeObj1.strftime('%M')) + '_' \
        + day2Padded0 + '_' + str(datetimeObj2.strftime('%H')) + '_' + str(datetimeObj2.strftime('%M'))

    return humanReadableInterval

def computeTimestamps(daysList):
    timestampList = []

    for day in daysList:
        transformedDateUTC = date(day[0], day[1], day[2])
        timestampStart = (transformedDateUTC.toordinal() - date(1970, 1, 1).toordinal()) * 24*60*60
        timestampEnd = (transformedDateUTC.toordinal() - date(1970, 1, 1).toordinal()) * 24*60*60 + 23*60*60 + 59*60 + 59
        timestampList.append((timestampStart, timestampEnd))

    return timestampList

intervalHeads = computeTimestamps([(2021, 11, 3), (2021, 11, 30)])        
 
intervalInSeconds = 60 * 60 * 5 # seconds (60) * minutes * hours
startTimestamp = intervalHeads[0][0] # first day, start timestamp
endTimestamp = intervalHeads[1][0] # second day, start timestamp

# n minutes intervals
timeIntervals = createTimeIntervals(startTimestamp, endTimestamp, intervalInSeconds)

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionFedora

allComments = list(db.comments.find()) + list(db.submissions.find())

# use a generic name for comments with no authors
genericAuthorName = 'JhonDoe25122020'

timestamps2Comments = {}
timestamps2RedditIds = {}
interactionsDict = {}
toInsert = {}

redditIds2Comments = dict(zip([comment['redditId'] for comment in allComments], [comment for comment in allComments]))

for comment in allComments:

    if (comment['author'] == False):
        comment['author'] = genericAuthorName

    createdTimestamp = int(comment['created'])

    if (createdTimestamp not in timestamps2Comments):
        timestamps2Comments[createdTimestamp] = []

    if (createdTimestamp not in timestamps2RedditIds):
        timestamps2RedditIds[createdTimestamp] = []
    
    timestamps2Comments[createdTimestamp].append(comment)
    
    # add parent too
    if ('parentRedditId' in comment):
        parentRedditId = comment['parentRedditId'].split('_')[1]

        if (parentRedditId not in timestamps2RedditIds[createdTimestamp]):
            timestamps2Comments[createdTimestamp].append(redditIds2Comments[parentRedditId])
            timestamps2RedditIds[createdTimestamp].append(parentRedditId)


for createdTimestamp in timestamps2Comments:

    if (createdTimestamp < startTimestamp or createdTimestamp > endTimestamp):
        continue

    associatedInterval = round((createdTimestamp - startTimestamp) / intervalInSeconds)

    if associatedInterval >= len(timeIntervals):
        continue

    dbName = convertTimestampIntervalToReadableInterval(timeIntervals[associatedInterval])

    if (dbName not in toInsert):
        toInsert[dbName] = []

    toInsert[dbName] += timestamps2Comments[createdTimestamp]

for dbName in toInsert:
    # open and close connection to avoid crashes
    dbClient = pymongo.MongoClient('localhost', 27017)
    db = dbClient.communityDetectionFedora

    print('Insert into ', dbName)
    noDuplicates = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in toInsert[dbName])]
    db[dbName].insert_many(noDuplicates)

    dbClient.close()
