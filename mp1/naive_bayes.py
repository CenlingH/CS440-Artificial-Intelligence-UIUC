# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023
# 好难啊

"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter  # 我就不用了


'''
util for printing values
'''


def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")


"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""


def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(
        trainingdir, testdir, stemming, lowercase, silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""


def naiveBayes(dev_set, train_set, train_labels, laplace=8, pos_prior=0.8, silently=False):
    print_values(laplace, pos_prior)
    trained_dict_pos, trained_dict_neg, num_pos, num_neg = train_dict(
        train_set, train_labels)
    prob_dict_pos, prob_UNK1 = smoothing(trained_dict_pos, num_pos, laplace)
    prob_dict_neg, prob_UNK2 = smoothing(trained_dict_neg, num_neg, laplace)
    yhats = []

    for doc in tqdm(dev_set):
        log_pos = math.log(pos_prior)
        log_neg = math.log(1-pos_prior)

        for word in doc:
            if word in prob_dict_pos:
                log_pos += prob_dict_pos[word]
            else:
                log_pos += prob_UNK1
            if word in prob_dict_neg:
                log_neg += prob_dict_neg[word]
            else:
                log_neg += prob_UNK2

        if log_pos > log_neg:
            yhats.append(1)
        else:
            yhats.append(0)
    print(len(yhats))
    '''for doc in tqdm(dev_set, disable=silently):
        yhats.append(-1)'''

    return yhats


'''
training phase:
    1. divide train_set into 2 dictionaries: positive and negative
    2. calculate P(word|positive) or P(word|negative),P(UNK|condition) using Laplace smoothing formula
'''


def train_dict(train_set, train_labels):
    # 1.create 2 dict, 1 for positive, 1 for negative
    dict_pos = {}
    dict_neg = {}
    num_pos = 0
    num_neg = 0
    # [[word1,word2,word3],[word1,word2,word3]]
    for i in range(len(train_set)):
        for word in train_set[i]:
            if train_labels[i] == 1:
                num_pos += 1
                if word in dict_pos:
                    dict_pos[word] += 1
                else:
                    dict_pos[word] = 1
            else:
                num_neg += 1
                if word in dict_neg:
                    dict_neg[word] += 1
                else:
                    dict_neg[word] = 1
    return dict_pos, dict_neg, num_pos, num_neg


def smoothing(dict_trained, count, laplace=8):
    prob_dict = {}
    V = len(dict_trained)  # number of word types
    for word in dict_trained:
        prob_dict[word] = math.log(
            (dict_trained[word]+laplace)/(count+laplace*(V+1)))
    prob_UNK = math.log(laplace/(count+laplace*(V+1)))
    return prob_dict, prob_UNK
