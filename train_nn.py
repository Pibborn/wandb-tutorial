from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_data(val_split=0.2):
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    x_train = train_df['excerpt'].to_list()
    y_train = train_df['target'].to_list()
    x_test = test_df['excerpt'].to_list()
    y_test = [0 for i in x_test]
    if val_split > 0:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(x_train, truncation=True, padding=True)
    if val_split > 0:
        val_encodings = tokenizer(x_val, truncation=True, padding=True)
    test_encodings = tokenizer(x_test, truncation=True, padding=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_encodings), y_train)
    )
    if val_split > 0:
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(val_encodings), y_val)
        )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(test_encodings), y_test)
    )
    if val_split > 0:
        return train_dataset, val_dataset, test_dataset
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_data, val_data, test_data = prepare_data()
    training_args = TFTrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1
    )
    with training_args.strategy.scope():
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

    trainer = TFTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_data,  # training dataset
        eval_dataset=val_data,  # evaluation dataset
    )

    trainer.train()
    y_pred = trainer.predict(test_data)[0]
    print(y_pred)



