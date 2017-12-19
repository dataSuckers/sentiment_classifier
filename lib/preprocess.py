from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
#from gensim.models import KeyedVectors
import fasttext

model = fasttext.skipgram('data.txt','model')
#en_model = KeyedVectors.load_word2vec_format('wiki.en.vec')
stopwords = set(stopwords.words("english"))

def tokenize(sentence):
    sentence = clean_sent(sentence)
    words = word_tokenize(sentence)
    word2 = []
    for w in words:
        if w not in stopwords:
            word2.append(w)
    return word2

def clean_sent(sentence):
    step1 = re.sub('RT ', '', sentence)
    step2 = re.sub('@[_A-Za-z0-9]+', '', step1)
    step3 = re.sub('#', '', step2)
    step4 = re.sub('http.+|via', '', step3)
    pat = r'\w?[.?\-",:!_~()]+'
    step5 = re.sub(pat, '', step4)
    step6 = re.sub('  +', ' ', step5)
    step7 = re.sub('\\n', '', step6)
    step8 = re.sub('^ | $', '', step7)

    return step8

def ave(word2):
    count = 0
    sum = 0
    for w in word2:
        if w in en_model:
            print('in',w)
            v = en_model[w]
            sum += v
            count += 1

    if count!=0:
        return sum / count


def pre_train_fasttext():
    thefile = open('data2.txt', 'w',encoding='utf-8')
    for i in merge:
        sentence = clean_sent(i)
        thefile.write("%s\n" % sentence)
    model = fasttext.skipgram(thefile, 'model')
    return model

def ave2(word2,model):
    count = 0
    sum = 0
    for w in word2:
        vec=model[w]
        sum+=vec
        count+=1

    return sum/count


def get_input(train,y):
    input = []
    y2 = []
    for i in range(len(train)):
        sentence=train[i]
        word_list = tokenize(sentence)
        print(word_list)
        inputave = ave(word_list)
        input.append(inputave)
        y2.append(y)
    return input, y2

model_ft= fasttext.load_model('W:/GWU3.0/nlp/project/wiki.en/wike.en.bin')#too big...

'''
######fasttext


words = []
for word in en_model.vocab:
    words.append(word)

print("Numer of Tokens:{}".format(len(words)))  # 2519370

print("Dimension of a word vector: {}".format(
    len(en_model[words[0]])
))  # 300

print("Vector components of a word: {}".format(
    en_model[words[0]]
))

find_similar_to = 'car'

'''