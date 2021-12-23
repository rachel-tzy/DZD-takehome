import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint, Callback, ProgbarLogger)
from tensorflow.keras import backend as K
import os

def step_size_decay(epoch):
    if epoch > 1 and epoch <= 30 and epoch%3==1:
        global step_size
        step_size = step_size/2
    return step_size

#Function to print epoch count, loss and step size (to observe decay) after every epoch
class FlushCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        print('Epoch %s: loss %s' % (epoch, logs.get('loss')))
        print("Step size:",K.eval(optimizer.lr))


seq_len = 310
max_epochs = 20     # Num of epochs training happens
mini_batch_size = 128
step_size = 0.05
model_name = 'cnn_codon1'
dir = 'model\\'+model_name

data = np.load("dataset.npy", allow_pickle=True).item()
ys = data["resistant"]  # [n_sample] (bool)

X_one_hot = np.load('data\\X_codon_one_hot.npy')
X_train, X_test, y_train, y_test = train_test_split(X_one_hot, ys, test_size=0.2, random_state=42)

model = load_model(dir+'\\model'+model_name+'.hdf5', compile=False)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
step_size_callback = LearningRateScheduler(step_size_decay)

# # Callbacks to save and retreive the best weight configurations found during training phase
all_callbacks = [step_size_callback, FlushCallback(),
                           ModelCheckpoint(dir+'\\model'+model_name+'.hdf5',
                                           save_best_only=True,
                                           verbose=1),
                           ProgbarLogger(count_mode='steps')]

#--------------- TRAINING ------------------#

hist = model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=max_epochs,
                 verbose=2, validation_split=0.2, callbacks=all_callbacks)
print(hist.history, file=open(dir+'\\history-'+model_name+'.txt', 'a'))
print(hist.history)
hist_df = pd.DataFrame(hist.history)
if os.path.exists(dir+'\\'+model_name+'.csv'):
    hist_df.to_csv(dir + '\\' + model_name + '.csv', mode='a', header=False, index=False)
else:
    hist_df.to_csv(dir + '\\' + model_name + '.csv', mode='a', header=True, index=False)
