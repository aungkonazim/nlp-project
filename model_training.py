import numpy as np
import pickle
from keras.utils import to_categorical
from random import randint
from numpy import array
from numpy import argmax
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import os

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(1, n_unique-1) for _ in range(length)]

def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


X_Y = pickle.load(open('data/X_Y.p','rb'))
encoded_docs = to_categorical(X_Y)
max_len = 6000
X_one_hot = encoded_docs[:max_len,:,:]
Y_one_hot = encoded_docs[max_len:,:,:]
embedding_matrix = pickle.load(open('data/embed_matrix.p','rb'))
word_index = pickle.load(open('data/word_index.p','rb'))
index_word = pickle.load(open('data/index_word.p','rb'))
vocab_size = len(word_index)+1
n_features = vocab_size
n_steps_in = 25
n_steps_out = 25

train, infenc, infdec = define_models(n_features, n_features, 128)
if os.path.isfile('data/my_model_weights.h5'):
    train.load_weights('data/my_model_weights.h5')
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
train.summary()
# train.fit([X_one_hot, Y_one_hot], Y_one_hot, epochs=1,batch_size=50)
# train.save_weights('data/my_model_weights.h5')
# train.save('data/model_final.h5')
# infdec.save('data/decoder_model.h5')
# infenc.save('data/enc_model.h5')
#
# latent_dim = 256
# num_encoder_tokens = 25
# num_decoder_tokens = 25
#
# encoder_inputs = Input(shape=(25,))
# x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
# x, state_h, state_c = LSTM(latent_dim,
#                            return_state=True)(x)
# encoder_states = [state_h, state_c]
#
# # Set up the decoder, using `encoder_states` as initial state.
# decoder_inputs = Input(shape=(25,))
# x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
# x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
# decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)
#
# # Define the model that will turn
# # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#
# # Compile & run training
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# # Note that `decoder_target_data` needs to be one-hot encoded,
# # rather than sequences of integers like `decoder_input_data`!
# # model.fit([X, Y], Y_one_hot,
# #           batch_size=64,
# #           epochs=1,
# #           validation_split=0.2)
# model.summary()


# input_context = Input(shape=(25,), dtype='int32', name='input_context')
# input_answer = Input(shape=(25,), dtype='int32', name='input_answer')
# LSTM_encoder = Bidirectional(LSTM(25, kernel_initializer="lecun_uniform"))
# LSTM_decoder = Bidirectional(LSTM(25, kernel_initializer="lecun_uniform"))
#
# Shared_Embedding = Embedding(output_dim=10, input_dim=vocab_size, weights=[embedding_matrix], input_length=25)
# # Shared_Embedding = Embedding(output_dim=25, input_dim=25, embeddings_initializer='identity')
#
# word_embedding_context = Shared_Embedding(input_context)
# context_embedding1 = LSTM_encoder(word_embedding_context)
# context_embedding = Dropout(0.5)(context_embedding1)
# word_embedding_answer = Shared_Embedding(input_answer)
# answer_embedding1 = LSTM_decoder(word_embedding_answer)
# answer_embedding = Dropout(0.5)(answer_embedding1)
#
# merge_layer = merge([context_embedding, answer_embedding], mode='concat', concat_axis=1)
# out = Dense(np.int32(vocab_size/2), activation="relu")(merge_layer)
#
# out2 = Dense(25, activation="linear")(out)
#
# model = Model(input=[input_context, input_answer], output = [out2])
#
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#
# model.summary()
#
# # model.fit([X, Y], Y, batch_size=64, epochs=2)
# # #
# # # # model.save('data/model.h5')
# # #
# # predicted = model.predict([X,Y])
# # #
# # print(np.shape(X),np.shape(Y),np.shape(predicted))
# #
# # print(predicted[0])