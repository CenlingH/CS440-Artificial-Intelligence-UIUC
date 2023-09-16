# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""
# 好难啊

import reader
import math
from tqdm import tqdm
# from collections import Counter


'''
utils for printing values
'''


def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")


def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")


"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""


def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(
        trainingdir, testdir, stemming, lowercase, silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""


def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.0005, bigram_laplace=0.006, bigram_lambda=0.4, pos_prior=0.99, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace,
                        bigram_lambda, pos_prior)
    yhats = []

    # a series of training
    trained_dict_pos, trained_dict_neg, num_pos, num_neg = train_dict(
        train_set, train_labels)
    trained_dict_pos_bi, trained_dict_neg_bi, num_pos_bi, num_neg_bi = bigram_dict(
        train_set, train_labels)

    prob_dict_pos, prob_UNK_pos = smoothing(
        trained_dict_pos, num_pos, unigram_laplace)
    prob_dict_neg, prob_UNK_neg = smoothing(
        trained_dict_neg, num_neg, unigram_laplace)
    prob_dict_pos_bi, prob_UNK_pos_bi = smoothing(
        trained_dict_pos_bi, num_pos_bi, bigram_laplace)
    prob_dict_neg_bi, prob_UNK_neg_bi = smoothing(
        trained_dict_neg_bi, num_neg_bi, bigram_laplace)

    for doc in tqdm(dev_set, disable=silently):
        log_pos_uni = math.log(pos_prior)
        log_neg_uni = math.log(1-pos_prior)
        log_pos_bi = math.log(pos_prior)
        log_neg_bi = math.log(1-pos_prior)

        for word in doc:
            if word in prob_dict_pos:
                log_pos_uni += prob_dict_pos[word]
            else:
                log_pos_uni += prob_UNK_pos
            if word in prob_dict_neg:
                log_neg_uni += prob_dict_neg[word]
            else:
                log_neg_uni += prob_UNK_neg

        for i in range(len(doc)-1):
            if (doc[i], doc[i+1]) in prob_dict_pos_bi:
                log_pos_bi += prob_dict_pos_bi[(doc[i], doc[i+1])]
            else:
                log_pos_bi += prob_UNK_pos_bi
            if (doc[i], doc[i+1]) in prob_dict_neg_bi:
                log_neg_bi += prob_dict_neg_bi[(doc[i], doc[i+1])]
            else:
                log_neg_bi += prob_UNK_neg_bi

        log_pos = (1-bigram_lambda)*log_pos_uni+bigram_lambda*log_pos_bi
        log_neg = (1-bigram_lambda)*log_neg_uni+bigram_lambda*log_neg_bi

        if log_pos > log_neg:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats


# training phase:

# use the train_dict in mp1, create {"word":count} for pos and neg
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


# very similar as train_dict, but this time we use bigram
def bigram_dict(train_set, train_labels):
    dict_pos_bi = {}
    dict_neg_bi = {}
    num_pos_bi = 0
    num_neg_bi = 0
    for i in range(len(train_set)):
        for j in range(len(train_set[i])-1):
            bi_1 = train_set[i][j]
            bi_2 = train_set[i][j+1]
            if train_labels[i] == 1:
                num_pos_bi += 1
                if (bi_1, bi_2) in dict_pos_bi:
                    dict_pos_bi[(bi_1, bi_2)] += 1
                else:
                    dict_pos_bi[(bi_1, bi_2)] = 1
            else:
                num_neg_bi += 1
                if (bi_1, bi_2) in dict_neg_bi:
                    dict_neg_bi[(bi_1, bi_2)] += 1
                else:
                    dict_neg_bi[(bi_1, bi_2)] = 1
    return dict_pos_bi, dict_neg_bi, num_pos_bi, num_neg_bi


# use smoothing function in mp1
def smoothing(dict_trained, count, laplace):
    prob_dict = {}
    V = len(dict_trained)  # number of word types
    for word in dict_trained:
        prob_dict[word] = math.log(
            (dict_trained[word]+laplace)/(count+laplace*(V+1)))
    prob_UNK = math.log((laplace+0.0005)/(count+laplace*(V+1)))
    return prob_dict, prob_UNK
