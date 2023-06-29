import numpy as np
import pandas as pd
import os
from pickle import dump, load

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from tensorflow.keras.preprocessing import image

def preprocess_diaries(X_diary_train, X_diary_test, embedding_dim):
    """
    Process Natural Language Meal Diaries
    """
    vocab_size = 50000
    max_length = 150
    trunc_type='post'
    padding_type='post'

    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(X_diary_train)

    training_sequences = tokenizer.texts_to_sequences(X_diary_train)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(X_diary_test)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

    X_diary_train = np.array(training_padded)
    X_diary_test = np.array(testing_padded)
    
    print('Found %s unique tokens.' % len(tokenizer.word_index))
    
    dump(tokenizer, open('tokenizer_64.pkl', 'wb'))
    return (X_diary_train, X_diary_test, tokenizer)

def load_images(df, root_dir):
    """
    Load Food Images
    """
    images = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = os.path.join(root_dir, row.category_id, str(index)+'.jpg')
        images.append(preprocess_image(path)[0])
    return np.array(images)

def preprocess_image(path):
    """
    Process Food Images
    """
    img = image.load_img(path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img
