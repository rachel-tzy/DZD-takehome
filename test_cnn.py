from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split

model_name = 'cnn_codon1'
dir = 'cv\\'+model_name
model = load_model(dir+'\\model'+model_name+'.hdf5', compile=False)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

data = np.load("dataset.npy", allow_pickle=True).item()
ys = data["resistant"]  # [n_sample] (bool)

X_codon_one_hot = np.load('data\\X_codon_one_hot.npy')
X_train, X_test, y_train, y_test = train_test_split(X_codon_one_hot, ys, test_size=0.2, random_state=42)


model.evaluate(X_test, y_test)
