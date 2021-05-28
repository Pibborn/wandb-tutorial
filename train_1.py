import pandas as pd
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# load data
df = pd.read_csv('data/train.csv')
corpus = df['excerpt']
y = df['target']
# print(corpus)
# print(target)
x = corpus.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

# preprocess
vectorizer = TfidfVectorizer(max_features=1000)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
x_val = vectorizer.transform(x_val)
print('Transformed tf-idf features: {}'.format(x_train.shape))

# train
model = SVR(C=1, kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_val)

# test
score = mean_squared_error(y_val, y_pred, squared=False)
print('Obtained score: {}'.format(score))



