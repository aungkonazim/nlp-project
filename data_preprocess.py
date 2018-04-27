
import pandas as pd
import numpy as np
import re
import pickle

# Load the data
lines = open('.\\data\\movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('.\\data\\movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


# limit = 0
# for i in range(limit, limit+5):
#     print(questions[i])
#     print(answers[i])
#     print()


# In[10]:

# Compare lengths of questions and answers
# print(len(questions))
# print(len(answers))


# In[11]:

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


# In[12]:

# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))


# In[13]:

# Take a look at some of the data to ensure that it has been cleaned well.
# limit = 110
# for i in range(limit, limit+5):
#     print(clean_questions[i])
#     print(clean_answers[i])
#     print()


# In[14]:

# Find the length of sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


# In[15]:

lengths.describe()


# In[16]:

print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))
#
#
# # In[17]:
#
# Remove questions and answers that are shorter than 2 words and longer than 20 words.
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

# for i in range(len(short_answers)):
#     short_answers[i] += ' end'
#     short_answers[i] = 'start ' + short_answers[i]
#
# for i in range(len(short_questions)):
#     short_questions[i] += ' end'
#     short_questions[i] = 'start ' + short_questions[i]


# In[18]:

print(short_questions[10],'-'*20,short_answers[10])
print("# of questions:", len(short_questions))
print("# of answers:", len(short_answers))
print("% of data used: {}%".format(round(len(short_questions)/len(questions),4)*100))

pickle.dump(short_questions,open('.\\data\\questions.p','wb'))
pickle.dump(short_answers,open('.\\data\\answers.p','wb'))

with open('data/question.txt','w') as f:
    for item in short_questions[:10000]:
        f.write("%s\n" % item)
f.close()

with open('data/answer.txt','w') as f:
    for item in short_answers[:10000]:
        f.write("%s\n" % item)
f.close()

