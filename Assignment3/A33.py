#!/usr/bin/env python
# coding: utf-8
import sys

if __name__ == "__main__":
    #input_path_1 = sys.argv[1]
    # In[2]:


    import pandas as pd
    import numpy as np

    data = open('pos.txt').read()
    parsed = data.split('\n')


    data_table2 = pd.DataFrame({
        "body_text": parsed[:-1]
    })
    data_table2.head()


    data1 = open('neg.txt').read()
    parsed = data.split('\n')
    import pandas as pd

    data_table1 = pd.DataFrame({
        "body_text": parsed[:-1]
    })

    data_table  = pd.concat([data_table1, data_table2], ignore_index=True, sort=False)


    print(data_table)


    # In[3]:


    import re
    punctuation_pattern = '|!|"|#|$|%|)|&|(|*|+|/|:|;|<|=|>|]|@|[(|\)|^|`|{|(|)|}|~|\t|\n'

    def remove_punct(text):
        text_nopunct = ''.join([(ch if ch not in punctuation_pattern else " ") for ch in text])
        return text_nopunct


    # In[4]:


    import re
    def tokenize(text):
        tokens = re.findall(r"[a-zA-Z0-9]+|[^\s]", text)
        return tokens


    # In[5]:


    import nltk
    data_table['withstopword'] = data_table['body_text'].apply(lambda x: tokenize(remove_punct(x.lower())))

    data_table.head()


    # In[8]:


    from gensim.models import Word2Vec
    import multiprocessing
    cores = multiprocessing.cpu_count()

    word2vec = Word2Vec(data_table['withstopword'], min_count=20,window=2,
                         size=300,
                         sample=6e-5, 
                         alpha=0.03, 
                         min_alpha=0.0007, 
                         negative=20,
                         workers=cores-1)
                         
    sim_words = word2vec.wv.most_similar('good',topn=20)  
    sim_words1 = word2vec.wv.most_similar('bad',topn=20)  
    print(sim_words)
    print(sim_words1)


    # In[ ]:





    # In[ ]:




