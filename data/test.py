from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os
import numpy as np
import pickle

path = "C:\\Users\\aungkon\\Desktop\\projects\\nlp-project-final\\data\\english"
files = os.listdir(path)
questions = []
answers = []



for file in files:
    with open(path+'\\'+file,'r',encoding='utf8') as f:
        data = f.readlines()
        a = True
        for i in range(3,len(data),1):
            if a:
                if data[i].find('- -') != -1:
                    questions.append(data[i].split('- -')[-1])
                    a=False
            else:
                if data[i].find('- -') == -1:
                    answers.append(data[i].split(' -')[-1])
                    a=True


for i,item in enumerate(questions):
    questions[i] = questions[i].split('\n')[0]
    answers[i] = answers[i].split('\n')[0]
    # print(questions[i],'---',answers[i])
    questions[i] += ' eos'
    answers[i] += ' eos'
combined_qa = questions+answers

t = Tokenizer()
t.fit_on_texts(combined_qa)
word_index = t.word_index
index_word = {v: k for k, v in word_index.items()}
vocab_size = len(word_index)+1
X_Y = t.texts_to_sequences(combined_qa)
X_Y = pad_sequences(X_Y, maxlen=10)
for k in X_Y[10]:
    if k in index_word.keys():
        print(index_word[k],end=' ')

X_Y_final = to_categorical(X_Y,num_classes=vocab_size)
X = X_Y_final[:len(questions),:,:]
Y = X_Y_final[len(questions):,:,:]
print(np.shape(X),np.shape(Y))
pickle.dump(X,open('data/X.p','wb'))
pickle.dump(Y,open('data/Y.p','wb'))
pickle.dump(word_index,open('data/word_index.p','wb'))
pickle.dump(t,open('data/tokenizer.p','wb'))





