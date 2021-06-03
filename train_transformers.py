import pandas as pd
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import wandb
from wandb.keras import WandbCallback
from keras import Sequential
from keras.layers import Dense, Activation


# wandb stuff
wandb.login()
wandb.init(project='wandb-tutorial', entity='Pibborn', config='defaults.yaml')

# argument loading
parser = argparse.ArgumentParser(description='Train a simple MLP model with tensorflow.')
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--activation', type=str, help='activation function', default='sigmoid')
parser.add_argument('--max_features', type=int, help='TFIDF features', default=100)
args = parser.parse_args()


# load data
df = pd.read_csv('data/train.csv')
corpus = df['excerpt']
y = df['target']
# print(df)
# print(corpus)
# print(target)
x = corpus.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16)

# preprocess
vectorizer = TfidfVectorizer(max_features=args.max_features)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
print('Transformed tf-idf features: {}'.format(x_train.shape))

# define MLP
model = Sequential()
model.add(Dense(100, input_dim=x_train.toarray().shape[1]))
model.add(Activation(args.activation))
model.add(Dense(100))
model.add(Activation(args.activation))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# train MLP
model.fit(x_train.toarray(), y_train, epochs=30, batch_size=128, validation_split=0.2, verbose=2,
          callbacks=[WandbCallback(log_gradients=True, training_data=(x_train.toarray(), y_train), save_model=True)])

y_pred = model.predict(x_test.toarray())
score = mean_squared_error(y_test, y_pred, squared=False)
print('Obtained score: {}'.format(score))
wandb.log({'rmse': score})


