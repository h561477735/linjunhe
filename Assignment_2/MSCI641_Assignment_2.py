#!/usr/bin/env python
# coding: utf-8

import sys

if __name__ == "__main__":
    input_path_1 = sys.argv[1]
    input_path_2 = sys.argv[2]
    input_path_3 = sys.argv[3]
    input_path_4 = sys.argv[4]
    input_path_5 = sys.argv[5]
    input_path_6 = sys.argv[6]


    # In[1]:


    import pandas as pd
    import numpy as np




    # train_pos = np.loadtxt("train_no_stopword_pos.csv", delimiter = '\n', dtype = 'str')
    # train_neg = np.loadtxt("train_no_stopword_neg.csv", delimiter = '\n', dtype = 'str')
    # val_pos = np.loadtxt("val_no_stopword_pos.csv", delimiter = '\n', dtype = 'str')
    # val_neg = np.loadtxt("val_no_stopword_neg.csv", delimiter = '\n', dtype = 'str')
    # test_pos = np.loadtxt("test_no_stopword_pos.csv", delimiter = '\n', dtype = 'str')
    # test_neg = np.loadtxt("test_no_stopword_neg.csv", delimiter = '\n', dtype = 'str')

    # In[2]:


    train_pos = np.loadtxt(input_path_1, delimiter = '\n', dtype = 'str')
    train_neg = np.loadtxt(input_path_2, delimiter = '\n', dtype = 'str')
    val_pos = np.loadtxt(input_path_3, delimiter = '\n', dtype = 'str')
    val_neg = np.loadtxt(input_path_4, delimiter = '\n', dtype = 'str')
    test_pos = np.loadtxt(input_path_5, delimiter = '\n', dtype = 'str')
    test_neg = np.loadtxt(input_path_6, delimiter = '\n', dtype = 'str')


    # In[3]:


    train_data = pd.DataFrame({'Label': ['Pos']*len(train_pos)+['Neg']*len(train_neg), 'Comment': np.concatenate((train_pos,train_neg))})


    # In[4]:


    val_data = pd.DataFrame({'Label': ['Pos']*len(val_pos)+['Neg']*len(val_neg), 'Comment': np.concatenate((val_pos,val_neg))})


    # In[5]:


    test_data = pd.DataFrame({'Label': ['Pos']*len(test_pos)+['Neg']*len(test_neg), 'Comment': np.concatenate((test_pos,test_neg))})


    # In[6]:


    train_data.head()


    # In[7]:


    import nltk
    ps = nltk.PorterStemmer()


    # In[8]:


    import re
    punctuation_pattern = '|!|"|#|$|%|)|&|(|*|+|/|:|;|<|=|>|]|@|[(|\)|^|`|{|(|)|}|~|\t|\n'
    def clean_text(text):
        text = "".join([word for word in text if word not in punctuation_pattern])
        tokens = re.split('\W+', text)
        text = " ".join([ps.stem(word) for word in tokens])
        return text

    train_data['text_nopunct'] = train_data['Comment'].apply(lambda x: clean_text(x))
    val_data['text_nopunct'] = val_data['Comment'].apply(lambda x: clean_text(x))
    test_data['text_nopunct'] = test_data['Comment'].apply(lambda x: clean_text(x))
    train_data.head()


    # In[9]:


    from sklearn.feature_extraction.text import CountVectorizer

    unigram_vect = CountVectorizer(ngram_range=(1,1))
    uni_counts = unigram_vect.fit_transform(train_data['text_nopunct'])
    uni_counts_val = unigram_vect.transform(val_data['text_nopunct'])
    uni_counts_test = unigram_vect.transform(test_data['text_nopunct'])


    # In[10]:


    bigram_vect = CountVectorizer(ngram_range=(2,2))
    bi_counts = bigram_vect.fit_transform(train_data['text_nopunct'])
    bi_counts_val = bigram_vect.transform(val_data['text_nopunct'])
    bi_counts_test = bigram_vect.transform(test_data['text_nopunct'])


    # In[11]:


    unibigram_vect = CountVectorizer(ngram_range=(1,2))
    unibi_counts = unibigram_vect.fit_transform(train_data['text_nopunct'])
    unibi_counts_val = unibigram_vect.transform(val_data['text_nopunct'])
    unibi_counts_test = unibigram_vect.transform(test_data['text_nopunct'])


    # In[12]:


    from sklearn.naive_bayes import MultinomialNB


    # In[13]:


    def train_NB(a,counts, counts_val):
        clf = MultinomialNB(alpha = a)
        clf_model = clf.fit(counts, train_data['Label'])
        y_pred = clf_model.predict(counts_val)
        print('Alpha: {}---- Accuracy: {}'.format(a, round((y_pred==val_data['Label']).sum() / len(y_pred), 3)))
        return (y_pred==val_data['Label']).sum() / len(y_pred)


    # In[14]:


    i = 0
    alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    accuracy = [None]*len(alpha)
    for a in alpha:
        accuracy[i] = train_NB(a,uni_counts,uni_counts_val)
        i = i+1
    best_alpha = alpha[accuracy.index(max(accuracy))]


    # In[15]:


    clf = MultinomialNB(alpha = best_alpha)
    clf_model_uni = clf.fit(uni_counts, train_data['Label'])
    y_pred = clf_model_uni.predict(uni_counts_test)
    print('Unigram model:---- Accuracy: {}'.format(round((y_pred==test_data['Label']).sum() / len(y_pred), 3)))


    # In[16]:


    i = 0
    accuracy = [None]*len(alpha)
    for a in alpha:
        accuracy[i] = train_NB(a,bi_counts,bi_counts_val)
        i = i+1
    best_alpha = alpha[accuracy.index(max(accuracy))]


    # In[17]:


    clf = MultinomialNB(alpha = best_alpha)
    clf_model_bi = clf.fit(bi_counts, train_data['Label'])
    y_pred = clf_model_bi.predict(bi_counts_test)
    print('Bigram model:---- Accuracy: {}'.format(round((y_pred==test_data['Label']).sum() / len(y_pred), 3)))


    # In[18]:


    i = 0
    accuracy = [None]*len(alpha)
    for a in alpha:
        accuracy[i] = train_NB(a,unibi_counts,unibi_counts_val)
        i = i+1
    best_alpha = alpha[accuracy.index(max(accuracy))]


    # In[19]:


    clf = MultinomialNB(alpha = best_alpha)
    clf_model_unibi = clf.fit(unibi_counts, train_data['Label'])
    y_pred = clf_model_unibi.predict(unibi_counts_test)
    print('Uni&bigram model:---- Accuracy: {}'.format(round((y_pred==test_data['Label']).sum() / len(y_pred), 3)))


    # In[ ]:





