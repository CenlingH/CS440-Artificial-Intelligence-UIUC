"""
Part 2: This is the simplest(no!) version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log
import pdb

# Note: remember to use these two elements when you find a probability is 0 in the training data.
# 这俩东西类似于alpha，其实本质上是一个预测（预测unk的具体数值）
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect
# alpha = 1e-5


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0)  # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(
        lambda: 0))  # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(
        lambda: 0))  # {tag0:{tag1: # }}

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    tagset = set()
    wordset = set()
    for s in sentences:
        for word, tag in s:
            if tag not in tagset:
                tagset.add(tag)
            if word not in wordset:
                wordset.add(word)

    # compute init_prob
    for s in sentences:
        init_prob[s[0][1]] += 1
    for i in init_prob:  # i是tag
        init_prob[i] /= len(sentences)

    # compute emit_prob
    for s in sentences:
        for (word_keys, tag_keys) in s:
            emit_prob[tag_keys][word_keys] += 1  # 现在有tag下每个word的数量
    for tag in tagset:
        if tag not in emit_prob:
            emit_prob[tag] = {}
    for tag_keys in emit_prob:
        V = len(emit_prob[tag_keys])
        n = sum(emit_prob[tag_keys].values())
        for word_keys in emit_prob[tag_keys]:
            emit_prob[tag_keys][word_keys] = (
                emit_prob[tag_keys][word_keys] + emit_epsilon) / (n+emit_epsilon * (V+1))
        emit_prob[tag_keys]['UNK'] = emit_epsilon / (n+emit_epsilon * (V+1))

    # compute trans_prob
    for s in sentences:
        for i in range(len(s)-1):
            trans_prob[s[i][1]][s[i+1][1]] += 1

    for tag in tagset:
        if tag not in trans_prob:
            trans_prob[tag] = {}

    for tag_pre_keys in trans_prob:
        V = len(trans_prob[tag_pre_keys])
        n = sum(trans_prob[tag_pre_keys].values())
        for tag_later_keys in trans_prob[tag_pre_keys]:
            trans_prob[tag_pre_keys][tag_later_keys] = trans_prob[tag_pre_keys][tag_later_keys] + \
                epsilon_for_pt / (n+epsilon_for_pt * (V+1))
        trans_prob[tag_pre_keys]['UNK'] = epsilon_for_pt / \
            (n+epsilon_for_pt * (V+1))
    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities 
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {}
    # This should store the log_prob for all the tags at current column (i)
    # This should store the tag sequence to reach each tag at column (i)
    predict_tag_seq = {}

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    if i == 0:
        for tag_now in prev_prob.keys():
            log_prob[tag_now] = prev_prob[tag_now] + \
                math.log(emit_prob[tag_now].get(
                    word, emit_prob[tag_now]['UNK']))
            predict_tag_seq[tag_now] = [tag_now]
        return log_prob, predict_tag_seq

    for tag_now in prev_prob.keys():
        v_max = 0
        tag_pre_max = ''
        for tag_pre, prob_pre in prev_prob.items():
            v = prob_pre+math.log(emit_prob[tag_now].get(word, emit_prob[tag_now]['UNK']))+math.log(
                trans_prob[tag_pre].get(tag_now, trans_prob[tag_pre]['UNK']))
            if v > v_max or v_max == 0:
                v_max = v
                tag_pre_max = tag_pre
        log_prob[tag_now] = v_max
        predict_tag_seq[tag_now] = prev_predict_tag_seq[tag_pre_max] + \
            [tag_now]
    return log_prob, predict_tag_seq


def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)

    predicts = []

    for sen in range(len(test)):
        sentence = test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(
                i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

            # print("111111111:", log_prob)
            # print("333333333:", predict_tag_seq)

        # TODO:(III)
        # according to the storage of probabilities and sequences, get the final prediction.

        max_tag = max(log_prob, key=log_prob.get)
        predicts.append([(sentence[i], predict_tag_seq[max_tag][i])
                         for i in range(length)])
    # print("5555555555:", max_tag)
    return predicts
