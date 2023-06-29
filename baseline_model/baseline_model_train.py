import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import datasets
from utils import models

from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, Dropout

from tensorflow.keras.callbacks import TensorBoard

############################################################
#  Load Data
############################################################
df = pd.read_csv('../data/meta.csv')
images = datasets.load_images(df, '../img_64')
embedding_dim = 100

train, test, X_image_train, X_image_test = train_test_split(df, images, test_size=0.20, random_state=2)
X_diary_train, X_diary_test, tokenizer = datasets.preprocess_diaries(train['diaries'], test['diaries'], embedding_dim)

############################################################
#  Process Dataset
############################################################
label = LabelEncoder()
y_train = label.fit_transform(train['category_id'])
y_test = label.transform(test['category_id'])
y_train = to_categorical(y_train, 64)
y_test = to_categorical(y_test, 64)

############################################################
#  Build Model
############################################################
tokenizer = load(open('tokenizer.pkl', 'rb'))

lstm = models.lstm_model(len(tokenizer.word_index) + 1, embedding_dim, X_diary_train.shape[1])
cnn = models.cnn_model(X_image_train.shape[1:])
concat = concatenate([lstm.output, cnn.output])
x = Dense(128, activation='relu')(concat)
x = Dropout(0.2)(x)
x = Dense(64, activation='softmax')(x)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)
model.summary()

############################################################
#  Train Model
############################################################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard_callback = TensorBoard(log_dir="logs/image/")

history = model.fit([X_diary_train, X_image_train], y_train, validation_data=([X_diary_test, X_image_test], y_test), epochs=50, batch_size=8, callbacks=[tensorboard_callback])
model.save('Multi_Food101_model.h5')

############################################################
#  Evaluation
############################################################
accr = model.evaluate([X_diary_test, X_image_test], y_test)
print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}'.format(accr[0],accr[1]))
