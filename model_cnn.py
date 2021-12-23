from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Convolution1D, Input, MaxPooling1D, Flatten, Dense, Dropout)
import os
import warnings
warnings.filterwarnings('ignore')

# Model construction for CNN
seq_len = 310 # Fixed length of a sequence (310 for codon seq, 1000 for some  char-level model,  910 for char-level model)
alph_size = 21 # 21 for codon-level model, 4 for char-level model

#--------------- CNN STRUCTURE ------------------#

#Input one-hot encoded sequence of chars
sequence_one_hot = Input(shape=(seq_len, alph_size), dtype='float32')

# 1st CNN layer without max-pooling
conv1 = Convolution1D(256, 1, activation='relu')(sequence_one_hot)

# 2nd CNN layer with max-pooling
conv2 = Convolution1D(256, 1, activation='relu')(conv1)
pool2 = MaxPooling1D(pool_size=3)(conv2)

# Reshaping to 1D array for further layers
flat = Flatten()(pool2)

# 1st fully connected layer with dropout
dense1 = Dense(1024, activation='relu')(flat)
dropout1 = Dropout(0.5)(dense1)

# 2nd fully connected layer with dropout
dense2 = Dense(1024,activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense2)

# 3rd fully connected layer with softmax outputs
dense3 = Dense(1, activation='sigmoid')(dropout2)
model = Model(sequence_one_hot, dense3)
model_name ='cnn_codon1'
dir = 'cv\\'+model_name
if os.path.exists(dir):
    model.save(dir+'\\model'+model_name+'.hdf5')
else:
    os.makedirs(dir)
    model.save(dir + '\\model' + model_name + '.hdf5')

print(model.summary())


with open(dir+'\\summary-'+model_name+'.txt', 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))