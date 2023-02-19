from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV as RSCV
import pandas as pd
import numpy as np
version = 0
with open('currentVersion.txt', 'r') as f:
    lines = f.readlines()
    version = int(lines[0])
trainingdata_path = "csv/training" + version.__str__() + ".csv"

df = pd.read_csv(trainingdata_path)
df["embedding"] = df.embedding.apply(eval).apply(
    np.array)  # convert string to array
df["relevant"] = df.relevant.apply(
    lambda x: 1 if x else 0)  # convert bool to int
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.relevant, test_size=0.2, random_state=42
)
param_grid = {'n_estimators': np.arange(50, 200, 15),
              'max_features': np.arange(0.1, 1, 0.1),
              'max_depth': [3, 5, 7, 9],
              'max_samples': [0.3, 0.5, 0.8],
              }

model = RSCV(RandomForestClassifier(), param_grid,
             n_iter=15).fit(X_train, y_train)
model = model.best_estimator_

print(model)
print(model.score(X_test, y_test))
