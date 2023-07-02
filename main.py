from typing import List
from clearScreen import cls
import openai
import pinecone
from decouple import config
from pyzotero import zotero
import pandas as pd
import time
import datetime
version = 0
with open('currentVersion.txt', 'r') as f:
    lines = f.readlines()
    version = int(lines[0])
trainingdata_path = "csv/training" + version.__str__() + ".csv"
trainingdata_path_newVersion = "csv/training" + (version+1).__str__() + ".csv"


def getCollectionId():
    if (config('ZOTERO_MAIN_COLLECTION_ID') != ""):
        return config('ZOTERO_MAIN_COLLECTION_ID')
    else:
        collections = zot.collections()
        # get all collection names
        collectionNames = ""
        for collection in collections:
            collectionNames += "\n- " + collection['data']['name']
        collectionNameInput = input(
            "Es existieren folgende Collections:" + collectionNames + "\nCollection Name: ")
        return getIdFromCollectionName(collectionNameInput, collections)


def getIdFromCollectionName(collectionName, collections):

    for collection in collections:
        if collection['data']['name'] == collectionName:
            return collection['data']['key']
    return None


openai.organization = config('OPENAI_ORG_ID')
openai.api_key = config('OPENAI_SECRET')
library_type = config('ZOTERO_LIBRARY_TYPE')
library_id = config('ZOTERO_USER_ID')
api_key = config('ZOTERO_KEY')
zot = zotero.Zotero(library_id, library_type, api_key)
main_collection_id = getCollectionId()
print(main_collection_id)
index = input("Insert start index (default= 0): ")
if index == "":
    index = 0
else:
    try:
        index = int(index)
    except:
        print("Invalid input, using default value")
        index = 0
limitInput = input("Insert limit - only in steps of 100s: ")
limitInput = int(limitInput)
limitInput = limitInput - 1
loops = (limitInput/100).__floor__()

items = zot.collection_items(main_collection_id, start=index)
currentIndex = index
while loops > 0:
    currentIndex += 100
    items += zot.collection_items(main_collection_id, start=currentIndex)
    loops -= 1
print(len(items))


class PaperInfo:
    def __init__(self, title, abstract, authors, id):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.embedding = list()

    def setEmbedding(self):
        model = "text-embedding-ada-002"
        paperInfoString = 'Title: %s \n\nAbstract: %s \n\nAuthors: %s' % (
            self.title, self.abstract, self.authors)
        try:
            response = openai.Embedding.create(
                model=model, input=paperInfoString)
        except:
            print("Rate limit exceeded, waiting 60 seconds")
            time.sleep(60)
            response = openai.Embedding.create(
                model=model, input=paperInfoString)

        # get embedding value from response that is in form of a json object. The embedding is in the "embedding" key inside the "data" key

        self.embedding = response['data'][0]['embedding']


paperList: List[PaperInfo] = []
skipped = 0
# print(items)
# we've retrieved the latest five top-level items in our library
# we can print each item's item type and ID
for item in items:
    authorListString = ""
    try:
        for author in item['data']['creators']:
            authorListString += author['firstName'] + \
                " " + author['lastName'] + ", "
    except:
        authorListString = "unknown"

    authorListString = authorListString.removesuffix(", ")
    title, abstract, id = "", "", ""
    try:
        title = item['data']['title']
    except:
        title = "unknown"
    try:
        abstract = item['data']['abstractNote']
    except:
        # skip this entry
        skipped += 1
        continue
    try:
        id = item['data']['key']
    except:
        skipped += 1
        continue
    paperInfo = PaperInfo(title, abstract, authorListString, id)
    paperList.append(paperInfo)
print("Found " + paperList.__len__().__str__() + " papers")
if (skipped > 0):
    print("Skipped " + skipped.__str__() +
          " papers because they didn't have an abstract")
convert = input("Start conversion? (y/n) ")
if convert != "y":
    exit()
# create embeddings for all papers at once to avoid rate limit
paperEmbeddings = list()
for paper in paperList:
    paperInfoString = 'Title: %s \n\nAbstract: %s \n\nAuthors: %s' % (
        paper.title, paper.abstract, paper.authors)
    paperEmbeddings.append(paperInfoString)

model = "text-embedding-ada-002"
response = None
try:
    response = openai.Embedding.create(model=model, input=paperEmbeddings)
except:
    print("Rate limit exceeded, waiting 60 seconds")
    time.sleep(60)
    response = openai.Embedding.create(model=model, input=paperEmbeddings)

    # get embedding value from response that is in form of a json object. The embedding is in the "embedding" key inside the "data" key
index = 0
for paper in paperList:
    paper.embedding = response['data'][index]['embedding']
    print("Created embedding for " + paper.title)
    index += 1
    # time.sleep(0.1)
    # progress = int((paperList.index(paper)+1)/paperList.__len__()*40)
    # cls()
    # print("Creating Embeddings: [" + "#" *
    #     int(progress) + " " * int(40-progress) + "]")
    # paper.setEmbedding()
    # paper.embedding = paper
print(paperList[0].embedding)
while True:
    action = input("Is this a training or a prediction? (t/p) ")
    if action == "t":
        relevace = input("Is this collection relevant? (y/n) ")
        relevant = "0"
        if relevace == "y":
            relevant = "1"
        date = datetime.datetime.now()
        df = pd.read_csv(trainingdata_path)
        for paper in paperList:
            print("Saving" + paper.title)
            new_row = pd.Series({'title': paper.title, 'abstract': paper.abstract,
                                'authors': paper.authors, 'embedding': paper.embedding, 'date': date, 'relevant': relevant})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        df.to_csv(trainingdata_path_newVersion, index=False, header=True)
        with open('currentVersion.txt', 'w') as f:
            f.write((version+1).__str__())
        print("success")
        break
    elif action == "p":
        date = datetime.datetime.now()

        df = pd.DataFrame({'title': 'title', 'abstract': 'abstract',
                           'authors': 'authors', 'embedding': 'embedding', 'date': 'date', 'id': 'id'}, index=[0])
        for paper in paperList:
            print("Saving " + paper.title)
            new_row = pd.Series({'title': paper.title, 'abstract': paper.abstract,
                                'authors': paper.authors, 'embedding': paper.embedding, 'date': date, 'id': paper.id})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        df.to_csv('csv/prediction.csv', index=False, header=False)
        break
