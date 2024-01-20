from scipy.io import loadmat
import numpy as np
import pandas as pd
import os

DATADIR = "competitionData/train"

for idx, data in enumerate(os.listdir(DATADIR)):
    print ("Source file: ", data)
    data_dict =  loadmat(os.path.join(DATADIR, data))
    for j in range(data_dict["sentenceText"].shape[0]):
        sentence = data_dict["sentenceText"][j]
        tx1 = data_dict["tx1"][0][j] #an array of something by 256
        tx2 = data_dict["tx2"][0][j]
        tx3 = data_dict["tx3"][0][j]
        tx4 = data_dict["tx4"][0][j]
        blockIdx = data_dict["blockIdx"][0][0]
        spikePow = data_dict["spikePow"][0][j]
    print (data_dict.keys())
    print ("tx1 shape: ", data_dict["tx1"].shape)
    print ("tx1 0 shape: ", data_dict["tx1"][0].shape)
    tx1 = data_dict["tx1"][0][0]
    print ("tx1 0 1 .shape: ", data_dict["tx1"][0][1].shape)
    print ("tx1 0 0 .shape: ", tx1.shape)
    print ("sentenceText: ", data_dict["sentenceText"].shape)
    print ("spikePow: ", data_dict["spikePow"].shape)

    sentenceText = data_dict["sentenceText"]
    tx2 = data_dict["tx2"][0][0]
    tx3 = data_dict["tx3"][0][0]
    tx4 = data_dict["tx4"][0][0]
    spikePow = data_dict["spikePow"][0][0]
    dataset = pd.DataFrame(sentenceText, columns=["sentence"])
    dataset["source"] = data
    if idx == 0:
        full_dataset = dataset
    else:
        full_dataset = pd.concat([full_dataset, dataset], axis=0)

print (dataset.head(5))
print (full_dataset.shape)
full_dataset["charLen"] = full_dataset["sentence"].apply(lambda x: len(x))
print (full_dataset["charLen"].max(), " chars at most in training set")
# these are the features in the dict: dict_keys(['__header__', '__version__', '__globals__', 'sentenceText', 'tx1', 'tx2', 'tx3', 'tx4', 'spikePow', 'blockIdx'])
# tx1 is arranged as a list of numpy arrays, one per sentenceText (i.e. data["tx1"][0][0] is the first sentenceText)
# this np array is  time_samples (remember 20 Hz samplig) x n_channels (256 electrodes)
# print (data["tx1"][0][0].shape)

# For example, t12.2022.07.29.mat contains sentenceText of length 200 (i.e. 200 sentences)
# so tx1 is a list of 200 np arrays. Each np array is of variable # rows, because the recording works at 20Hz, but
# has the same num of columns (256) because the recording is always done with 256 electrodes