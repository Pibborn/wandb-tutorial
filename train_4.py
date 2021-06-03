import pandas as pd
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import wandb

# wandb stuff
wandb.login()
wandb.init(project='wandb-tutorial', entity='Pibborn', config='defaults.yaml')

# argument loading
parser = argparse.ArgumentParser(description='Train a simple SVR model.')
parser.add_argument('--C', type=float, help='SVR slack', default=1.0)
parser.add_argument('--max_features', type=int, help='TFIDF features', default=500)
parser.add_argument('--kernel', type=str, help='SVR kernel', default='linear')
args = parser.parse_args()

wandb.config.update(args, allow_val_change=True)

# load data
df = pd.read_csv('data/train.csv')
corpus = df['excerpt']
y = df['target']
# print(df)
# print(corpus)
# print(target)
x = corpus.to_numpy()
df_test = pd.read_csv('data/test.csv')
x_test = df['excerpt']

# preprocess
vectorizer = TfidfVectorizer(max_features=wandb.config.max_features)
x_train = vectorizer.fit_transform(x)
x_test = vectorizer.transform(x_test)
print('Transformed tf-idf features: {}'.format(x_train.shape))

# train
model = SVR(C=wandb.config.C, kernel=wandb.config.kernel)
model.fit(x_train, y)
y_pred = model.predict(x_train)

# test
score = mean_squared_error(y, y_pred, squared=False)
print('Obtained score: {}'.format(score))
wandb.log({'rmse': score})

result = pd.DataFrame(df['id'], columns=['id'])
y_pred = pd.DataFrame(y_pred, columns=['result'])
result = pd.concat([result, y_pred], axis=1)
result.to_csv('output.csv', index=False)
