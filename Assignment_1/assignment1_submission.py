import sys

if __name__ == "__main__":
    input_path = sys.argv[1]

    
    data = open(input_path).read()
    parsed = data.split('\n')
    import pandas as pd
    data_table = pd.DataFrame({
        "body_text": parsed[:-1]
    })
    import re
    def tokenize(text):
        tokens = re.findall(r"[a-zA-Z0-9]+|[^\s]", text)
        return tokens
    
    import re
    punctuation_pattern = '|!|"|#|$|%|)|&|(|*|+|/|:|;|<|=|>|]|@|[(|\)|^|`|{|(|)|}|~|\t|\n'

    def remove_punct(text):
        text_nopunct = ''.join([(ch if ch not in punctuation_pattern else " ") for ch in text])
        return text_nopunct
    #data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))

    data_table['withstopword'] = data_table['body_text'].apply(lambda x: tokenize(remove_punct(x.lower())))
    import nltk
    stopword = nltk.corpus.stopwords.words('english')

    def remove_stopwords(tokenized_list):
        text = [word for word in tokenized_list if word not in stopword]
        return text
    data_table['withoutstopword'] = data_table["withstopword"].apply(lambda x: remove_stopwords(x))
    import pandas
    import numpy as np

    data_split = data_table.sample(frac=1).reset_index(drop=True)

    #print(len(data_split))
    train_len = len(data_split) * 4//5
    val_len = len(data_split) * 1//10
    tes_len= len(data_split) * 1//10
    #print(train_len,val_len,tes_len)

    train_data = data_split[:train_len]
    validation_data = data_split [train_len:train_len+val_len]
    test_data = data_split[train_len+val_len:]
    #print(train_data)
    import numpy as np

    train_list = train_data['withstopword'].tolist()
    val_list = validation_data['withstopword'].tolist()
    test_list = test_data['withstopword'].tolist()
    train_list_no_stopword = train_data['withoutstopword'].tolist()
    val_list_no_stopword = validation_data['withoutstopword'].tolist()
    test_list_no_stopword = test_data['withoutstopword'].tolist()
    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')
    np.savetxt("train_no_stopword.csv", train_list_no_stopword,delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,delimiter=",", fmt='%s')
