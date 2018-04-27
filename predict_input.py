import numpy as np
import pickle
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
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
    return np.array(output)
t = pickle.load(open('data/tokenizer.p','rb'))
word_index = pickle.load(open('data/word_index.p','rb'))
index_word = pickle.load(open('data/index_word.p','rb'))
vocab_size = len(word_index)+1
n_features = vocab_size
n_steps_in = 25
n_steps_out = 25

X_Y = pickle.load(open('data/X_Y.p','rb'))
# input_sentence = ['who are you']
# processed_input_sentence = []
# for i in range(len(input_sentence)):
#     processed_input_sentence.append('start '+input_sentence[i]+' end')

# X_Y = t.texts_to_sequences(processed_input_sentence)
# X_Y = t.texts_to_sequences(X_Y[0])
# X_Y = pad_sequences(X_Y, maxlen=25)
input_encoder = to_categorical(X_Y[1:2],num_classes=n_features)

print(np.shape(input_encoder))
model = load_model('data/model_final.h5')
infenc = load_model('data/enc_model.h5')
infdec = load_model('data/decoder_model.h5')


predicted = predict_sequence(infenc, infdec,input_encoder, n_steps_out, n_features)

print(one_hot_decode(predicted))

