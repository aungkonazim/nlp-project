from utils import read_glove_vecs
import pickle
from pprint import pprint
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_len = 6000
questions = list(pickle.load(open('.\\data\\questions.p','rb')))[:max_len]
answers = list(pickle.load(open('.\\data\\answers.p','rb')))[:max_len]
combined_qa = questions+answers

embeddings_index = {}
glove_data = 'data/glove.6B.50d.txt'
f = open(glove_data,'r',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    value = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = value
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
t = Tokenizer()
t.fit_on_texts(combined_qa)
embedding_dimension = 10
word_index = t.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(str(word).lower())
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector[:embedding_dimension]

pickle.dump(embedding_matrix,open('data/embed_matrix.p','wb'))

X_Y = t.texts_to_sequences(combined_qa)
X_Y = pad_sequences(X_Y, maxlen=25)

X = X_Y[:len(questions)]
Y = X_Y[len(questions):]
pickle.dump(X,open('data/X.p','wb'))
pickle.dump(Y,open('data/Y.p','wb'))
pickle.dump(X_Y,open('data/X_Y.p','wb'))



index_word = {v: k for k, v in word_index.items()}



pickle.dump(word_index,open('data/word_index.p','wb'))
pickle.dump(index_word,open('data/index_word.p','wb'))
pickle.dump(t,open('data/tokenizer.p','wb'))

