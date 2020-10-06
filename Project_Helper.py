def ProjectPath():
    return 'C:/Users/eabea/Dropbox/Documents/ISYE6740/Project'

def TextProcess(DataSet, ExtraFeatures,Embedding,max_words,max_len):
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from Project_Helper import ProjectPath
    import pandas as pd
    import os


    os.chdir(ProjectPath() + '/Data')
    DataSet = pd.read_csv(DataSet)
    Sentences = []

    for i in range(DataSet.shape[0]):
        Sentences.append(DataSet.iloc[i][0])

    if Embedding:

        OneHotSentences = [one_hot(s, max_words) for s in Sentences]
        X = pad_sequences(OneHotSentences, maxlen=max_len, padding='post')

    else:
        t = Tokenizer(num_words=max_words)
        t.fit_on_texts(Sentences)

        X = pd.DataFrame((t.texts_to_matrix(Sentences, mode='count')))

    if ExtraFeatures:
        X[str(max_words)] = DataSet.iloc[:, 3]
        X[str(max_words + 1)] = DataSet.iloc[:, 4]
        X[str(max_words + 2)] = DataSet.iloc[:, 5]

    return X, DataSet.iloc[:, 6], max_words
