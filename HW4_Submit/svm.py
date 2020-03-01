
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Read Data
df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(5))

# Cleaning text data
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)
print(df.head())

# Data set for training and testing
X_train = df.loc[:10000, 'review'].values
y_train = df.loc[:10000, 'sentiment'].values
X_test = df.loc[10000:, 'review'].values
y_test = df.loc[10000:, 'sentiment'].values


# Processing documents into tokens
porter = PorterStemmer()
def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Download the nltk package
nltk.download('stopwords')
stop = stopwords.words('english')

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


# Training a SVM model for document classification

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [str.split] ,
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [str.split],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

svm_tfidf = Pipeline([('vect', tfidf),
                     ('clf', SVC(random_state=1))])

gs_svm_tfidf = GridSearchCV(svm_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

# Fit and Performance Measurement
if __name__ == '__main__':
    gs_svm_tfidf.fit(X_train, y_train)
    print('Best parameter set: %s ' % gs_svm_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_svm_tfidf.best_score_)
    clf = gs_svm_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
