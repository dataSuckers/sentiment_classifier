# -*- coding: utf-8 -*-

import pandas as pd
import utils

file=pd.read_csv('Sentiment.csv')
text=file['text']#len=13817
sent=file['sentiment']#get rid of neutural take 0 as negative, take 1 as postive
'''
list=[]
for t in text:
    t= re.sub('^RT ','',t)
    list.append(t)
'''
#get rid of neutural take 0 as negative, take 1 as postive
text1=[]
sent1=[]
for i in range(len(sent)):
    if sent[i]!='Neutral':
        text1.append(text[i])
        if sent[i]=='Positive':
            sent1.append(1)
        else:
            sent1.append(0)
text2 = []
sent2 = []
text3 = []
sent3 = []
file2 = pd.read_csv('k2.csv',encoding = "ISO-8859-1")
text2 = file2['text'].tolist()#len=6918
sent2 = file2['sent'].tolist()

file3 = pd.read_csv('s2.csv',encoding = "ISO-8859-1")
text3 = file3['SentimentText'].tolist()#len=1048575
sent3 = file3['Sentiment'].tolist()

merge=[]
mergesent=[]
merge=text1+text2+text3
mergesent=sent1+sent2+sent3


####################
pos=[]
neg=[]
for i in range(len(mergesent)):
    if mergesent[i]==1:
        pos.append(merge[i])#pos len =560701
    else:
        neg.append(merge[i])# neg len = 505689

#TODO: clean up 'RT', website address, punctuation

dict_total_pos={}
for sentence in pos:
    token_list = utils.tokenize(sentence)
    mydict = utils.create_word_features(token_list)
    dict_total_pos.update(mydict)

dict_total_neg = {}
for sentence in neg:
    token_list = utils.tokenize(sentence)
    mydict = utils.create_word_features(token_list)
    dict_total_neg.update(mydict)



