from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import random
from tqdm import tqdm
from babel.dates import format_date
import matplotlib.pyplot as plt
import pickle
import os
from keras.preprocessing.sequence import pad_sequences

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

def one_step_attention(a, s_prev):
    """ 
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights 
    "alphas" and the hidden states "a" of the Bi-LSTM. 
     
    Arguments: 
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a) 
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s) 
     
    Returns: 
    context -- context vector, input of the next (post-attetion) LSTM cell 
    """

    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a,s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas,a])
    ### END CODE HERE ###

    return context

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):

        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs = [X, s0, c0], outputs = outputs)

    ### END CODE HERE ###

    return model



X = pickle.load(open('data/X.p','rb'))
Y = pickle.load(open('data/Y.p','rb'))
word_index = pickle.load(open('data/word_index.p','rb'))
index_word = {v: k for k, v in word_index.items()}
t = pickle.load(open('data/tokenizer.p','rb'))
Tx = 10
Ty = 10
m = X.shape[0]
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(word_index)+1, activation=softmax)


model = model(Tx, Ty, n_a, n_s, len(word_index)+1, len(word_index)+1)

opt = Adam(lr=0.008, beta_1=0.9, beta_2=0.999,decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.summary()


s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Y.swapaxes(0,1))

if os.path.isfile("C:\\Users\\aungkon\\Desktop\\projects\\nlp-project-final\\data\\data\\attention_model_final.h5"):
    # model = load_model('data/attention_model_final.h5')
    model.fit([X, s0, c0], outputs, epochs=20, batch_size=10)
    model.save('data/attention_model_final.h5')
else:
    model.fit([X, s0, c0], outputs, epochs=5, batch_size=10)
    model.save('data/attention_model_final.h5')

# EXAMPLES = ['hello eos']
# X_Y = t.texts_to_sequences(EXAMPLES)
# X_Y = pad_sequences(X_Y, maxlen=10)
# X_Y_final = to_categorical(X_Y, num_classes=len(word_index)+1)
prediction = model.predict([X, s0, c0])
prediction = np.array(prediction).swapaxes(0,1)
print(np.shape(prediction))
prediction = np.argmax(prediction, axis = -1)
for k in range(prediction.shape[0]):
    for j in prediction[k,:]:
        print(j,end=' ')
    print()
# output = [index_word[int(i)] for i in prediction if int(i) in index_word.keys()]
# print("source:", EXAMPLES[0])
# print("output:", ''.join(output))


print(index_word)







