# This is a sample Python script.
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

model = load_model("model_weights/model_9.h5")
# model._make_predict_function()

model_tmp = ResNet50(weights="imagenet", input_shape=(224,224,3))

# Create a new model by removing the last layer from the model
model_res = Model(model_tmp.input,model_tmp.layers[-2].output)

def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_img(img):
    img = preprocess_image(img)
    feature_vector = model_res.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector

with open("loads/idx_to_word.pkl","rb") as i2w:
    idx_to_word = pickle.load(i2w)

with open("loads/word_to_idx.pkl","rb") as w2i:
    word_to_idx = pickle.load(w2i)

max_len = 35

def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()  # Word with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)

        if word == "endseq":
            break

    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption



# load and prepare the photograph
def image_caption(image):
    enc = encode_img(image)
    # generate description
    description = predict_caption(enc)
    # description = generate_desc(model, tokenizer, photo, max_length)
    print(description)
    return description
