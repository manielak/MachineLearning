import pandas as pd
from sklearn.dummy import DummyClassifier          
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_predict_model(X, y, model, success_metric=accuracy_score):
    model.fit(X, y)
    y_pred = model.predict(X)
    
    print("Distribution:")
    print( pd.Series(y_pred).value_counts() )
    
    return success_metric(y, y_pred)
	
	
vowels = ['a', 'ą', 'e', 'ę', 'i', 'o', 'u', 'y']

def how_many_vowels(name):
    count = sum( map(lambda x: int(x in vowels), name.lower()) )
    
    return count

#how_many_vowels('Jana')

df['count_vowels'] = df['name'].map(how_many_vowels)
train_and_predict_model(df[['len_name', 'count_vowels'] ], y, LogisticRegression(solver='lbfgs'))


model = LogisticRegression(solver='lbfgs')


df['last_letter'] = df['name'].map(lambda x: x[-1])
df['last_letter_cnt'] = df['last_letter'].factorize()[0]

X = df[['len_name', 'count_vowels', 'first_is_vowel', 'last_letter_cnt'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs'))


def last_is_vowel(name):
    return name.lower()[-1] in vowels

#last_is_vowel('Ada') and confirm

df['last_is_vowel'] = df['name'].map(last_is_vowel)

X = df[['last_is_vowel'] ]
train_and_predict_model(X, y, LogisticRegression(solver='lbfgs', max_iter=200))



accuracy_score(y, y_pred)

df['gender_pred'] = y_pred
df['gender_pred'].value_counts()


