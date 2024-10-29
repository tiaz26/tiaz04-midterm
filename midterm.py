import pickle 
import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
import nltk.metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
import numpy as np

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


trainingSet = pd.read_csv('./data/train.csv')
testingSet = pd.read_csv('./data/test.csv')


# trainingSet['Score'] = trainingSet['Score']
# print('train.csv shape is : ', trainingSet.shape)
# print('test.csv shapes is : ', testingSet.shape)
# print()
# trainingSet['Score'].value_counts().plot(kind='bar', alpha=.5)
# plt.show()


    
def user(df):
    numbers = df.groupby('UserId').agg({
        'Score': ['mean', 'std', 'count']
    }).reset_index()
    numbers.columns = ['mean', 'std', 'review','UserId']
    numbers['std'] = numbers['std'].fillna(0)
    df = df.merge(numbers, on='UserId', how='left')
    
    return df

def calc_sentiment(text):
    if isinstance(text, str):
        return sia.polarity_scores(text)['compound']
    else:
        return 0
    
def add_features_to(df):
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    df['Text_Length'] = df['Text'].str.len()
    df['Text_Length'] = df['Text_Length'].fillna(0)

    df['Sentiment'] = df['Text'].apply(calc_sentiment)
    df['Sentiment_Summary'] = df['Summary'].apply(calc_sentiment)

    df = user(df)
    
    df['mean'] = df['mean'].fillna(df['mean'].mean())
    df['std'] = df['std'].fillna(df['std'].mean())
    df['review'] = df['review'].fillna(1)

    return df

if exists('./data/X_train2.csv'):
    X_train = pd.read_csv('./data/X_train2.csv')
if exists('./data/X_submission2.csv'):
    X_submission = pd.read_csv('./data/X_submission2.csv')
else:
    train = add_features_to(trainingSet)

    X_submission = pd.merge(train, testingSet, left_on='Id', right_on='Id', how='right')

    X_submission = X_submission.drop(columns=['Score_x'])
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})

    X_train = train[train['Score'].notnull()]

    X_submission.to_csv('./data/X_submission.csv', index=False)
    X_train.to_csv('./data/X_train.csv', index=False)


"""splitting data for train and test"""

X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(columns=['Score']),
    X_train['Score'],
    test_size=0.25,
    random_state=0
)

"""feature selection"""

features = [
    'Sentiment', 
    'Sentiment_Summary', 
    'HelpfulnessNumerator', 
    'HelpfulnessDenominator', 
    'Text_Length',
    'mean',
    'std',
    'review'
]

X_train_select = X_train[features]
X_test_select = X_test[features]
X_submission_select = X_submission[features]

"""model xgboost"""

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train_select, Y_train)

Y_test_predictions = model.predict(X_test_select)

print('Accuracy:', accuracy_score(Y_test, Y_test_predictions))

"""confusion matrix"""

plt.figure(figsize=(10,8))
conf_matrix = confusion_matrix(Y_test, Y_test_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


"""submission"""

X_submission['Score'] = model.predict(X_submission_select)
X_submission['Score'] = X_submission['Score']
submission = X_submission[['Id', 'Score']]
submission.to_csv('./data/submission3.csv', index=False)