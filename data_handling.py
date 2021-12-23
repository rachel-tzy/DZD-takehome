import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from Bio.Seq import Seq
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
def dna_to_num(s, fixed_length=1000):
    alphabet = {'A':1, 'C':2, 'G':3, 'T':4}
    result = np.zeros(fixed_length)
    for i, char in enumerate(s):
        if i >= fixed_length:
            break
        result[i] = alphabet[char]
    return result

def one_hot_dna(data,fixed_length=1000):
    result = []
    for seq in data:
        result.append(to_categorical(dna_to_num(seq, fixed_length)))
    return np.array(result)

def one_hot_label(data):
    result = []
    for label in data:
        if label:
            result.append([1,0])
        else:
            result.append([0,1])
    return np.array(result)

def lstm_tokenizer(data, MAX_SEQUENCE_LENGTH):
    tokenizer = Tokenizer(num_words=None, char_level=True)
    tokenizer.fit_on_texts(['ACGT'])
    X = tokenizer.texts_to_sequences(data)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return X

def one_hot_protein(data):
    result = []
    for seq in data:
        result.append(to_categorical(codon_to_num(seq)))
    return np.array(result)

def codon_to_num(s, fixed_length=310):
    codon_dict = np.load('codons_aa.npy', allow_pickle=True).item()
    codon_to_number = {}
    i = 0
    for codon in codon_dict:
        codon_to_number[codon] = i
        i = i + 1
    result = np.zeros(fixed_length)
    for i, char in enumerate(s):
        result[i] = codon_to_number[char]
    return result

def transform_to_codon(data):
    n = len(data)
    translated_result = []
    translated_result_one_hot = []
    for i in range(n):
        coding_dna = Seq(xs[i])
        translated = coding_dna.translate()
        translated_result.append(translated)
        translated_result_one_hot.append(to_categorical(codon_to_num(translated)))
    return np.array(translated_result), np.array(translated_result_one_hot)



def NT_statistics(data_one_hot):
    sta = np.sum(data_one_hot, axis=1)
    return sta

if __name__ == '__main__':
    # load data
    data = np.load("dataset.npy", allow_pickle=True).item()
    xs = data["genes"]  # [n_sample, arbitrary length string object] 10000 samples
    ys = data["resistant"]  # [n_sample] (bool)
    # n_sample = xs.shape[0]
    # X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)
    # X_one_hot_800 = one_hot_dna(xs, fixed_length=800)[:, :, 1:5]
    # np.save('data\\X_one_hot_800.npy', X_one_hot_800)

    # X_one_hot_600 = np.load('data\\X_one_hot_800.npy')
    # print(X_one_hot_600.shape)
    # X_test_one_hot = np.load('data\\X_test_one_hot.npy')
    # y_test_seq = np.load('data\\Y_test_seq.npy')
    # print(X_test_one_hot[0])
    # print(X_test_one_hot.shape)
    # X_seq_910 = lstm_tokenizer(xs, 910)
    # np.save('data//X_seq_910.npy', X_seq_910)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, ys, test_size=0.20, random_state=42)

    # sta_test = np.sum(X_test_one_hot, axis=1)
    # group0_index = np.where(y_test_seq[0:1000] == 0)
    # group1_index = np.where(y_test_seq[0:1000] == 1)
    # plt.scatter(group0_index, sta_test[group0_index, 0], c='r', s=1)
    # plt.scatter(group1_index, sta_test[group1_index, 0], c='b', s=1)
    # plt.show()
    # # print(sta.shape)
    # # print(sta)
    # translated, translated_one_hot = transform_to_codon(xs)
    # translated_test, translated_one_hot_test = transform_to_codon(X_test)
    # np.save('data//X_codon.npy', translated)
    # np.save('data//X_test_codon.npy', translated_test)
    # np.save('data//X_codon_one_hot.npy', translated_one_hot)
    # np.save('data//X_test_codon_one_hot.npy', translated_one_hot_test)
    # X_codon = np.load('data//X_codon.npy', allow_pickle=True)
    count = 0
    for i in range(100000):
        if xs[i][0:3] != 'ATG':
            count = count + 1
    print(count/100000)

