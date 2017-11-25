# -*- coding: utf-8 -*-


import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
#Prepare feature vectors per training mail and its labels

kf = StratifiedKFold(n_splits=10)

totalsvm = 0
totalNB = 0
totalMatSvm = np.zeros((2,2));
totalMatNB = np.zeros((2,2));
def tf_idf(clean_text,mergesent):
    for train_index, test_index in kf.split(clean_text,mergesent):
        X_train = [clean_text[i] for i in train_index]
        X_test = [clean_text[i] for i in test_index]
        y_train = [mergesent[i] for i in train_index]
        y_test = [mergesent[i] for i in test_index]
        #y_train, y_test = mergesent[train_index], mergesent[test_index]
        #vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
        #train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
        #test_corpus_tf_idf = vectorizer.transform(X_test)
        #vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
        vectorizer2 = TfidfVectorizer(analyzer='char',min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
        train_corpus_tf_idf = vectorizer2.fit_transform(X_train) 
        test_corpus_tf_idf = vectorizer2.transform(X_test)
        model1 = LinearSVC()
        model2 = MultinomialNB()
        model1.fit(train_corpus_tf_idf,y_train)
        model2.fit(train_corpus_tf_idf,y_train)
        
        result1 = model1.predict(test_corpus_tf_idf)
        result2 = model2.predict(test_corpus_tf_idf)
        totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1, labels=[0,1])
        totalMatNB = totalMatNB + confusion_matrix(y_test, result2, labels=[0,1])
        totalsvm = totalsvm+sum(y_test==result1)
        totalNB = totalNB+sum(y_test==result2)
        
    print ("Confusion matrix for SVM",totalMatSvm)    
    print ("True positives - ", totalsvm)
    print ("Confusion matrix for Naive Bayes classifier\n",totalMatNB)
    print ("True positives - ",totalNB)
    print (classification_report(y_test, result1))
    print (classification_report(y_test, result2))
    return model1,model2

import random

def compare_the_wrong_one(model1,model2):
    test_sample = random.sample(range(1,len(clean_text)),1000)
    X_test = [clean_text[i] for i in test_sample]
    X_test_idf = vectorizer2.transform(X_test)
    y_test = [mergesent[i] for i in test_sample]
    result_sample_svm = model1.predict(X_test_idf)
    result_sample_NB = model2.predict(X_test_idf)
    
    diff_svm=0
    wrong_predict_svm=[]
    wrong_predict_NB=[]
    diff_NB=0
    for i in range(1000):
        if y_test[i]!=result_sample_svm[i]:
            idx = test_sample[i]
            diff_svm+=1#167
            wrong_predict_svm.append([X_test[i],result_sample_svm[i],y_test[i],merge[idx]])
        if y_test[i]!=result_sample_NB[i]:
            idx = test_sample[i]
            diff_NB+=1#248
            wrong_predict_NB.append([X_test[i],result_sample_NB[i],y_test[i],merge[idx]])
            
    return wrong_predict_svm,wrong_predict_NB
        
#error analysis