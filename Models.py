def M_2048x512(vocab_size,embed_dim,max_len):
    from tensorflow import keras
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense

    model = keras.Sequential()
    model.add(Input(shape = vocab_size))
    model.add(Dense(2048,activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    return model

def M_4096(vocab_size,embed_dim,max_len):
    from tensorflow import keras
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense

    model = keras.Sequential()
    model.add(Input(shape = vocab_size))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    return model

def M_Embedding30(vocab_size,embed_dim,max_len):
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Flatten

    model = keras.Sequential()
    model.add(Embedding(vocab_size,embed_dim,input_length=max_len))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def M_Embedding_LSTM128x256(vocab_size,embed_dim,max_len):
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Embedding


    model = keras.Sequential(vocab_size,embed_dim,max_len)
    model.add(Embedding(vocab_size, embed_dim, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def M_Embedding_GRU128x256(vocab_size,embed_dim,max_len):
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import GRU
    from tensorflow.keras.layers import Embedding

    model = keras.Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=max_len))
    model.add(GRU(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def M_Embedding_LSTM128x256DropOut(vocab_size,embed_dim,max_len):
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Dropout

    model = keras.Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=max_len))
    model.add(LSTM(128 ,dropout = 0.3 ,recurrent_dropout = 0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def M_Embedding_GRU128x256DropOut(vocab_size,embed_dim,max_len):
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import GRU
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import Dropout

    model = keras.Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=max_len))
    model.add(GRU(128,dropout = 0.3 ,recurrent_dropout = 0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model