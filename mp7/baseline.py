"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    dic_baseline = defaultdict(lambda: defaultdict(
        lambda: 0))
    tag_count = defaultdict(lambda: 0)
    for sentence in train:
        for word, tag in sentence:
            dic_baseline[word][tag] += 1
            tag_count[tag] += 1
    most_tag = max(tag_count, key=tag_count.get)  # 最大值对应的键
    ret = []
    for sentence in test:
        r = []
        for word in sentence:
            if word not in dic_baseline:
                r.append((word, most_tag))
            else:
                r.append((
                    word, max(dic_baseline[word], key=dic_baseline[word].get)))
        ret.append(r)
    return ret
