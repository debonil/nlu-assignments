# %% [markdown]
# ## Natural Language Understanding || Assignment 2
#
# Submitted by:
# - Debonil Ghosh (M21AIE225)
# - Ravi Shankar Kumar (M21AIE247)
# - Saurav Chowdhury (M21AIE256)

# %% [markdown]
# Question 1 : Releation classifier

# %%
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier


# %% [markdown]
# Data Preparation

# %%
import json

data = []
with open('data/train.json', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# %%
passages = []
relations = []
print(f'Original document count = {len(data)}')
for d in data:
    for p in d['passages']:
        for f in p['facts']:
            passages.append(f['annotatedPassage'])
            relations.append(f['propertyId'])
print(f'Total fact count = {len(passages)}')

# %%

# Vectorize using Bag of Words
vectorizer = CountVectorizer()
text_vectors = vectorizer.fit_transform(passages)

# Concatenate subject and object vectors
X = text_vectors
y = np.array(relations)

# %% [markdown]
# Split data into training and testing sets

# %%

X_train, X_test, y_train, y_test = train_test_split(
    passages, y, test_size=0.2, random_state=42)

# %% [markdown]
# #### Train the classifier

# %%
# Train the classifier
# nltk.download()
stemmer = SnowballStemmer("english", ignore_stopwords=False)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer()
clf = Pipeline([('vect', stemmed_count_vect),
                ('mnb', RandomForestClassifier(n_estimators=100, random_state=42)),
                ])

# %%
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# %% [markdown]
# Evaluation part

# %%
relations_df = pd.read_csv('data/relations.csv')
classes = relations_df['propertyName'].values
classes


# %%


def confusionMatrixAndAccuracyReport(Y_test, Y_pred, classes, title=''):
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = metrics.accuracy_score(Y_test, Y_pred)

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    f1_score = metrics.f1_score(Y_test, Y_pred, average='weighted')

    plt.figure(figsize=(15, 15))
    plt.title(
        f'{title} : Accuracy : {overallAccuracy*100:3.2f}% | F1 Score : {f1_score*100:3.2f}% ', size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues',
                fmt='g', xticklabels=classes, yticklabels=classes)

    plt.savefig(f'results/confusion_mat_{title}.png', bbox_inches='tight')
    plt.show()
    print(f'Accuracy: {overallAccuracy*100:3.3f}%')
    print(f'F1 Score: {f1_score*100:3.3f}%')
    classwiseAccuracy_df = pd.DataFrame(
        data=[classwiseAccuracy], columns=classes)
    print(
        f'\nClasswise Accuracy Score: \n{classwiseAccuracy_df.to_markdown(index=False)}')
    print('\nConfusion Matrix:')
    print(cm.to_markdown())
    return overallAccuracy


# %%

# Evaluate the classifier
y_pred = clf.predict(X_test)
print(f'Predicted class => {y_pred[10]}')
print(f'accuracy_score :: {metrics.accuracy_score(y_test, y_pred):0.3f}')
print(f'f1_score ::{metrics.f1_score(y_test, y_pred,average="weighted"):0.3f}')
print(metrics.classification_report(y_test, y_pred))
confusionMatrixAndAccuracyReport(
    y_test, y_pred, classes, title='Relation_Classifier')


# %% [markdown]
# ### Saving model  for future use

# %%
dump(clf, 'models/relation_classifer.joblib')
