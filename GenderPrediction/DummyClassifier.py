import pandas as pd
from sklearn.dummy import DummyClassifier          
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("../input/polish_names.csv")
df.head()

df['length'] = df['name'].map(lambda x: len(x))
df.sample(10)

X = df[ ['length'] ].values
y = df['target'].values

model = DummyClassifier()
model.fit(X, y)
y_pred = model.predict(X)

df['gender_pred'] = y_pred
df['gender_pred'].value_counts()

df[ df.target != y_pred ].shape # błędna odpowiedź
accuracy_score(y, y_pred)
