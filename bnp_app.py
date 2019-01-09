# -*- coding: utf-8 -*-
# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl  --data input_word="good" http://localhost:5000/predict
#   curl  --data input_word="am" http://localhost:5000/predict
#   curl  --data input_word="bad" http://localhost:5000/predict

# import the necessary packages
import numpy as np
import flask
import io
import tensorflow as tf
import os
import itertools
import codecs
import re
import json
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import argparse

from src.TextImageGenerator import TextImageGenerator
from src.utils import *

parser = argparse.ArgumentParser("bnp_app")
parser.add_argument("--weight_path", type=str, default="./model/weights24.h5", help="path for the model weight")
args = parser.parse_args()



# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


np.random.seed(55)



def build_model(weight_path):
    img_w = 128
    # Input Parameters
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))
    
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    minibatch_size = 32
    
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
    
    fdir = os.path.dirname(get_file('wordlists.tgz',
                                    origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))
    
    img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                 bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words
                                 )
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    
    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    
    # transforms RNN output to character activations:
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.load_weights(weight_path)
    # captures output of softmax so we can decode the output during visualization
    #test_func = K.function([input_data], [y_pred])
    
    #global model_p
    model_p = Model(inputs=input_data, outputs=y_pred)
    #global graph
    graph = tf.get_default_graph()
    return model_p, graph


def decode_predict_ctc(out, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
      beam_width = top_paths
    for i in range(top_paths):
      lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                           greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
      text = labels_to_text(lables)
      results.append(text)
    return results
  




def predict_a_image(mdoel, a, top_paths = 1):
  c = np.expand_dims(a.T, axis=0)
  net_out_value = mdoel.predict(c)
  top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
  return top_pred_texts


model_p, graph = build_model(args.weight_path)

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {}

    if flask.request.method == "POST":
        if flask.request.get_data():
            input_word = flask.request.form.get('input_word', '')
            h = 64
            w = 128
            a = paint_text(input_word,h = h, w = w)
            b = a.reshape((h, w))
            c = np.expand_dims(a.T, axis=0)

            with graph.as_default():
                net_out_value = model_p.predict(c)
                pred_texts = decode_predict_ctc(net_out_value)
                top_3_paths = predict_a_image(model_p, a, top_paths = 3)

            data["pred_texts"] = pred_texts
            data["top_3_paths"] = top_3_paths

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host='0.0.0.0')
