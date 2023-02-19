# imports

import os
import pandas as pd
import numpy as np
import datetime
from pyzotero import zotero
from decouple import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from clearScreen import cls


def createZot():
    library_type = "user"
    library_id = config('ZOTERO_USER_ID')
    api_key = config('ZOTERO_KEY')
    return zotero.Zotero(library_id, library_type, api_key)


cls()
print("Starting classification...")
version = 0
with open('currentVersion.txt', 'r') as f:
    lines = f.readlines()
    version = int(lines[0])
# load data
trainingdata_path = "csv/training" + version.__str__() + ".csv"
prediction_path = "csv/prediction.csv"
cls()
print("Loading data...")
df = pd.read_csv(trainingdata_path)
df["embedding"] = df.embedding.apply(eval).apply(
    np.array)  # convert string to array
df["relevant"] = df.relevant.apply(
    lambda x: 1 if x else 0)  # convert bool to int
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.relevant, test_size=0.2, random_state=42
)

# train random forest classifier
cls()
print("Training classifier...")
clf = RandomForestClassifier(max_depth=5, max_features=0.8, max_samples=0.5,  # type: ignore
                             n_estimators=170)
clf.fit(X_train, y_train)
df = pd.read_csv(prediction_path)
df["embedding"] = df.embedding.apply(eval).apply(
    np.array)  # convert string to array
cls()
print("Predicting relevance...")
relevancePrediction = clf.predict(list(df.embedding.values))
# for i in range(len(relevancePrediction)):
#    if relevancePrediction[i] == 1:
#       print(i.__str__() + "- relevant: " + df["title"][i])
#   else:
#       print(i.__str__() + "- not relevant: " + df["title"][i])
date = datetime.datetime.now()
save = False
inMenu = True
i = -1
inPointMenu = True
renderAbstract = False
while inMenu:
    inPointMenu = True
    if (i < len(relevancePrediction)-1):
        i += 1
    page = i+1
    indexString = page.__str__() + "/" + len(relevancePrediction).__str__()
    if (page < 10):
        indexString = "0" + indexString
    while inPointMenu:
        cls()
        print("ENTER to continue, a to show abstract, c to change prediction, s to save, q to quit, p for previous")
        print("------------ "+indexString+" ------------")
        if relevancePrediction[i] == 1:
            print('\033[92m' + df["title"][i] + '\033[0m')
        else:
            print(df["title"][i])
        if renderAbstract:
            print("")
            print(df["abstract"][i])
            renderAbstract = False
        action = input()
        if action == "a":
            print(df["abstract"][i])
            renderAbstract = True
        elif action == "":
            inPointMenu = False
        elif action == "c":
            prevPred = relevancePrediction[i]
            if (prevPred == 1):
                relevancePrediction[i] = 0
            else:
                relevancePrediction[i] = 1
        elif action == "s":
            action = input("Are you sure you want to save? (y/n)")
            if action == "y":
                save = True
                inMenu = False
                inPointMenu = False
                break
        elif action == "q":
            action = input("Are you sure you want to quit? (y/n)")
            if action == "y":
                inMenu = False
                inPointMenu = False
                break
        elif action == "p":
            if i > 0:
                i -= 1
        elif action == "e":
            # add item to zotero collection via its id
            zot = createZot()
            for x in range(len(relevancePrediction)):
                if relevancePrediction[x] == 1:
                    id = df["id"][x]
                    item = zot.item(id)
                    updated = zot.add_tags(item, ["relevant"])
                    print(item)
                    print(updated)

        elif action == "ep":
            zot = createZot()
            x = 0
            while x <= i:
                if relevancePrediction[x] == 1:
                    id = df["id"][x]
                    zot.add_tags(id, ["relevant"])
                    print("marked " + df["title"][x] + " as relevant")
                x += 1

if save:
    cls()
    print("Saving...")
    trainingData = pd.read_csv(trainingdata_path)
    for index in range(len(relevancePrediction)):
        new_row = pd.Series({'title': df["title"][index], 'abstract': df["abstract"][index],
                            'authors': df["authors"][index], 'embedding': list(df["embedding"][index]), 'date': date, 'relevant': relevancePrediction[index]})
        trainingData = pd.concat(
            [trainingData, new_row.to_frame().T], ignore_index=True)
    trainingData.to_csv('csv/training' + (version+1).__str__() + '.csv',
                        index=False, header=True)
    with open('currentVersion.txt', 'w') as f:
        f.write((version+1).__str__())


preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)
report = classification_report(y_test, preds)


print(report)
