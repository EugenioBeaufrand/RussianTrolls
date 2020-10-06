from Models import M_2048x512,M_4096,M_Embedding30,M_Embedding_LSTM128x256,M_Embedding_GRU128x256,M_Embedding_LSTM128x256DropOut,M_Embedding_GRU128x256DropOut
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from Project_Helper import TextProcess
import tensorflow as tf

class MyCallBack(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= 0.985:
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True

DataSet = 'DataSet.csv'
ExtraFeatures = True #Set True for User feature inclusion
Embedding = False #Set True for Text Embedding instead of OneHot
max_words = 5000
pad_sequences_to = 20
embed_dim = 30

X,Y,max_words = TextProcess(DataSet,ExtraFeatures,Embedding,max_words,pad_sequences_to)

x_train,x_test,y_train,y_test = train_test_split(X , Y, test_size = 0.2)

callback = MyCallBack()

with tf.device("/device:GPU:0"):
    model = M_2048x512(x_train.shape[1],embed_dim,pad_sequences_to)
    history = model.fit(x_train,y_train,batch_size=1024,epochs= 100,callbacks = [callback],validation_data=(x_test, y_test))

loss0, acc0 = model.evaluate(x_train, y_train, verbose=0)
loss1, acc1 = model.evaluate(x_test, y_test, verbose=0)

print('Train Accuracy: %.3f' % acc0)
print('Test Accuracy: %.3f' % acc1)

num_epochs = len(history.history['accuracy'])

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.axis([0,num_epochs,0.7,1])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.axis([0,num_epochs,0,1])
plt.show()

print(tf.math.confusion_matrix(y_test,model.predict_classes(x_test)))