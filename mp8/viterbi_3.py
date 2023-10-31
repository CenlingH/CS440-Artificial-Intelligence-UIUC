import math
from collections import defaultdict
from math import log


# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect
# alpha = 1e-5
# 虽然不知道那个epsilon到底怎么算吧，但我觉得我写的逻辑上是这个逻辑，跟viterbi_2的大思路是一样的，具体数字不仔细算了，准确率上去就ok


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
    # wordset = set()
    word_tag_dict = defaultdict(lambda: defaultdict(lambda: 0))
    tag_word_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for s in sentences:
        for word, tag in s:
            tagset.add(tag)
            # wordset.add(word)
            word_tag_dict[word][tag] += 1
            tag_word_dict[tag][word] += 1

    # compute init_prob
    for s in sentences:
        init_prob[s[0][1]] += 1
    for i in init_prob:
        init_prob[i] /= len(sentences)

    # compute emit_prob
    # viterbi_2 improve emit_prob(compute hapax):
    hapax_dict = defaultdict(lambda: 0)  # hapax_dict是tag出现次数
    for word in word_tag_dict.keys():
        if sum(word_tag_dict[word].values()) == 1:
            key = list(word_tag_dict[word].keys())[0]
            hapax_dict[key] += 1
    sum_hapax = sum(hapax_dict.values())
    for tag in hapax_dict:
        hapax_dict[tag] /= sum_hapax

    # viterbi_3 新加的
    fix_prob_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for word in word_tag_dict.keys():
        if sum(word_tag_dict[word].values()) == 1:

            # 超长警告！1
            # est,ion,ful,ed,ly,er,ant,th,ship,ing,ment,ness 后缀
            # able,ise,en,ive,or,ness,less,ity,s 后缀补充
            # a,be,de,dis,ex,in,mis,non,over,pre,re,uni,with 前缀
            if word[-3:] == 'est':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_est'] += 1
            elif word[-3:] == 'ion':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ion'] += 1
            elif word[-3:] == 'ful':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ful'] += 1
            elif word[-2:] == 'ed':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ed'] += 1
            elif word[-2:] == 'ly':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ly'] += 1
            elif word[-2:] == 'er':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_er'] += 1
            elif word[-3:] == 'ant':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ant'] += 1
            elif word[-2:] == 'th':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_th'] += 1
            elif word[-4:] == 'ship':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ship'] += 1
            elif word[-3:] == 'ing':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ing'] += 1
            elif word[-4:] == 'ment':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ment'] += 1
            elif word[-4:] == 'ness':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ness'] += 1
            # 后缀补充
            elif word[-4:] == 'able':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_able'] += 1
            elif word[-3:] == 'ise':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ise'] += 1
            elif word[-2:] == 'en':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_en'] += 1
            elif word[-3:] == 'ive':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ive'] += 1
            elif word[-2:] == 'or':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_or'] += 1
            elif word[-4:] == 'ness':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ness'] += 1
            elif word[-4:] == 'less':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_less'] += 1
            elif word[-3:] == 'ity':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_ity'] += 1
            elif word[-1:] == 's':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['end_s'] += 1
            # 前缀开始
            elif word[:1] == 'a':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_a'] += 1
            elif word[:2] == 'be':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_be'] += 1
            elif word[:2] == 'de':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_de'] += 1
            elif word[:3] == 'dis':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_dis'] += 1
            elif word[:2] == 'ex':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_ex'] += 1
            elif word[:2] == 'in':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_in'] += 1
            elif word[:3] == 'mis':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_mis'] += 1
            elif word[:3] == 'non':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_non'] += 1
            elif word[:4] == 'over':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_over'] += 1
            elif word[:3] == 'pre':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_pre'] += 1
            elif word[:2] == 're':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_re'] += 1
            elif word[:3] == 'uni':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_uni'] += 1
            elif word[:4] == 'with':
                tag = list(word_tag_dict[word].keys())[0]
                fix_prob_dict[tag]['start_with'] += 1

    for tag in fix_prob_dict.keys():
        for fix_type in fix_prob_dict[tag].keys():
            fix_prob_dict[tag][fix_type] /= (hapax_dict[tag]*sum_hapax)
    # compute
    emit_prob = tag_word_dict
    for tag in tagset:
        if tag not in emit_prob:
            emit_prob[tag] = {}
    for tag_keys in emit_prob:
        if tag_keys in hapax_dict:
            emit_epsilon_new = emit_epsilon*hapax_dict[tag_keys]
        else:
            emit_epsilon_new = emit_epsilon*1e-5
        V = len(emit_prob[tag_keys])
        n = sum(emit_prob[tag_keys].values())
        for word_keys in emit_prob[tag_keys]:
            emit_prob[tag_keys][word_keys] = (
                emit_prob[tag_keys][word_keys] + emit_epsilon_new) / \
                (n+emit_epsilon_new * (V+1))

        # 超长警告！2
        if 'end_est' in fix_prob_dict[tag_keys]:
            epsilon_est = emit_epsilon_new*fix_prob_dict[tag_keys]['end_est']
        else:
            epsilon_est = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_est'] = epsilon_est / \
            (n+epsilon_est * (V+1))
        if 'end_ion' in fix_prob_dict[tag_keys]:
            epsilon_ion = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ion']
        else:
            epsilon_ion = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ion'] = epsilon_ion / \
            (n+epsilon_ion * (V+1))
        if 'end_ful' in fix_prob_dict[tag_keys]:
            epsilon_ful = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ful']
        else:
            epsilon_ful = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ful'] = epsilon_ful / \
            (n+epsilon_ful * (V+1))
        if 'end_ed' in fix_prob_dict[tag_keys]:
            epsilon_ed = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ed']
        else:
            epsilon_ed = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ed'] = epsilon_ed / \
            (n+epsilon_ed * (V+1))
        if 'end_ly' in fix_prob_dict[tag_keys]:
            epsilon_ly = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ly']
        else:
            epsilon_ly = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ly'] = epsilon_ly / \
            (n+epsilon_ly * (V+1))
        if 'end_er' in fix_prob_dict[tag_keys]:
            epsilon_er = emit_epsilon_new*fix_prob_dict[tag_keys]['end_er']
        else:
            epsilon_er = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_er'] = epsilon_er / \
            (n+epsilon_er * (V+1))
        if 'end_ant' in fix_prob_dict[tag_keys]:
            a_epsilonnt = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ant']
        else:
            a_epsilonnt = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ant'] = a_epsilonnt / \
            (n+a_epsilonnt * (V+1))
        if 'end_th' in fix_prob_dict[tag_keys]:
            epsilon_th = emit_epsilon_new*fix_prob_dict[tag_keys]['end_th']
        else:
            epsilon_th = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_th'] = epsilon_th / \
            (n+epsilon_th * (V+1))
        if 'end_ship' in fix_prob_dict[tag_keys]:
            epsilon_ship = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ship']
        else:
            epsilon_ship = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ship'] = epsilon_ship / \
            (n+epsilon_ship * (V+1))
        if 'end_ing' in fix_prob_dict[tag_keys]:
            epsilon_ing = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ing']
        else:
            epsilon_ing = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ing'] = epsilon_ing / \
            (n+epsilon_ing * (V+1))
        if 'end_ment' in fix_prob_dict[tag_keys]:
            epsilon_ment = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ment']
        else:
            epsilon_ment = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ment'] = epsilon_ment / \
            (n+epsilon_ment * (V+1))
        if 'end_ness' in fix_prob_dict[tag_keys]:
            epsilon_ness = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ness']
        else:
            epsilon_ness = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ness'] = epsilon_ness / \
            (n+epsilon_ness * (V+1))
        # 后缀补充
        if 'end_able' in fix_prob_dict[tag_keys]:
            epsilon_able = emit_epsilon_new*fix_prob_dict[tag_keys]['end_able']
        else:
            epsilon_able = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_able'] = epsilon_able / \
            (n+epsilon_able * (V+1))
        if 'end_ise' in fix_prob_dict[tag_keys]:
            ise_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ise']
        else:
            ise_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ise'] = ise_epsilon / \
            (n+ise_epsilon * (V+1))
        if 'end_en' in fix_prob_dict[tag_keys]:
            en_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_en']
        else:
            en_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_en'] = en_epsilon / \
            (n+en_epsilon * (V+1))
        if 'end_ive' in fix_prob_dict[tag_keys]:
            ive_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ive']
        else:
            ive_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ive'] = ive_epsilon / \
            (n+ive_epsilon * (V+1))
        if 'end_or' in fix_prob_dict[tag_keys]:
            or_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_or']
        else:
            or_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_or'] = or_epsilon / \
            (n+or_epsilon * (V+1))
        if 'end_ness' in fix_prob_dict[tag_keys]:
            ness_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ness']
        else:
            ness_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ness'] = ness_epsilon / \
            (n+ness_epsilon * (V+1))
        if 'end_less' in fix_prob_dict[tag_keys]:
            less_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_less']
        else:
            less_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_less'] = less_epsilon / \
            (n+less_epsilon * (V+1))
        if 'end_ity' in fix_prob_dict[tag_keys]:
            ity_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_ity']
        else:
            ity_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_ity'] = ity_epsilon / \
            (n+ity_epsilon * (V+1))
        if 'end_s' in fix_prob_dict[tag_keys]:
            s_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['end_s']
        else:
            s_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['end_s'] = s_epsilon / \
            (n+s_epsilon * (V+1))
        # 前缀开始
        if 'start_a' in fix_prob_dict[tag_keys]:
            a_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_a']
        else:
            a_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_a'] = a_epsilon / \
            (n+a_epsilon * (V+1))
        if 'start_be' in fix_prob_dict[tag_keys]:
            be_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_be']
        else:
            be_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_be'] = be_epsilon / \
            (n+be_epsilon * (V+1))
        if 'start_de' in fix_prob_dict[tag_keys]:
            de_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_de']
        else:
            de_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_de'] = de_epsilon / \
            (n+de_epsilon * (V+1))
        if 'start_dis' in fix_prob_dict[tag_keys]:
            dis_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_dis']
        else:
            dis_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_dis'] = dis_epsilon / \
            (n+dis_epsilon * (V+1))
        if 'start_ex' in fix_prob_dict[tag_keys]:
            ex_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_ex']
        else:
            ex_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_ex'] = ex_epsilon / \
            (n+ex_epsilon * (V+1))
        if 'start_in' in fix_prob_dict[tag_keys]:
            in_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_in']
        else:
            in_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_in'] = in_epsilon / \
            (n+in_epsilon * (V+1))
        if 'start_mis' in fix_prob_dict[tag_keys]:
            mis_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_mis']
        else:
            mis_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_mis'] = mis_epsilon / \
            (n+mis_epsilon * (V+1))
        if 'start_non' in fix_prob_dict[tag_keys]:
            non_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_non']
        else:
            non_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_non'] = non_epsilon / \
            (n+non_epsilon * (V+1))
        if 'start_over' in fix_prob_dict[tag_keys]:
            over_epsilon = emit_epsilon_new * \
                fix_prob_dict[tag_keys]['start_over']
        else:
            over_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_over'] = over_epsilon / \
            (n+over_epsilon * (V+1))
        if 'start_pre' in fix_prob_dict[tag_keys]:
            pre_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_pre']
        else:
            pre_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_pre'] = pre_epsilon / \
            (n+pre_epsilon * (V+1))
        if 'start_re' in fix_prob_dict[tag_keys]:
            re_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_re']
        else:
            re_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_re'] = re_epsilon / \
            (n+re_epsilon * (V+1))
        if 'start_uni' in fix_prob_dict[tag_keys]:
            uni_epsilon = emit_epsilon_new*fix_prob_dict[tag_keys]['start_uni']
        else:
            uni_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_uni'] = uni_epsilon / \
            (n+uni_epsilon * (V+1))
        if 'start_with' in fix_prob_dict[tag_keys]:
            with_epsilon = emit_epsilon_new * \
                fix_prob_dict[tag_keys]['start_with']
        else:
            with_epsilon = emit_epsilon_new*1e-5
        emit_prob[tag_keys]['start_with'] = with_epsilon / \
            (n+with_epsilon * (V+1))

        unk_epsilon = emit_epsilon_new-epsilon_est-epsilon_ion
        emit_prob[tag_keys]['UNK'] = unk_epsilon / \
            (n+unk_epsilon * (V+1))

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
            trans_prob[tag_pre_keys][tag_later_keys] = (
                trans_prob[tag_pre_keys][tag_later_keys] + epsilon_for_pt) / (n+epsilon_for_pt * (V+1))
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
            '''log_prob[tag_now] = prev_prob[tag_now] + \
                math.log(emit_prob[tag_now].get(
                    word, emit_prob[tag_now]['UNK']))'''

            if word in emit_prob[tag_now]:
                emit_p = emit_prob[tag_now][word]
            else:
                # 超长警告！3
                if word[-3:] == 'est':
                    emit_p = emit_prob[tag_now]['end_est']
                elif word[-3:] == 'ion':
                    emit_p = emit_prob[tag_now]['end_ion']
                elif word[-3:] == 'ful':
                    emit_p = emit_prob[tag_now]['end_ful']
                elif word[-2:] == 'ed':
                    emit_p = emit_prob[tag_now]['end_ed']
                elif word[-2:] == 'ly':
                    emit_p = emit_prob[tag_now]['end_ly']
                elif word[-2:] == 'er':
                    emit_p = emit_prob[tag_now]['end_er']
                elif word[-3:] == 'ant':
                    emit_p = emit_prob[tag_now]['end_ant']
                elif word[-2:] == 'th':
                    emit_p = emit_prob[tag_now]['end_th']
                elif word[-4:] == 'ship':
                    emit_p = emit_prob[tag_now]['end_ship']
                elif word[-3:] == 'ing':
                    emit_p = emit_prob[tag_now]['end_ing']
                elif word[-4:] == 'ment':
                    emit_p = emit_prob[tag_now]['end_ment']
                elif word[-4:] == 'ness':
                    emit_p = emit_prob[tag_now]['end_ness']
                # 后缀补充
                elif word[-4:] == 'able':
                    emit_p = emit_prob[tag_now]['end_able']
                elif word[-3:] == 'ise':
                    emit_p = emit_prob[tag_now]['end_ise']
                elif word[-2:] == 'en':
                    emit_p = emit_prob[tag_now]['end_en']
                elif word[-3:] == 'ive':
                    emit_p = emit_prob[tag_now]['end_ive']
                elif word[-2:] == 'or':
                    emit_p = emit_prob[tag_now]['end_or']
                elif word[-4:] == 'ness':
                    emit_p = emit_prob[tag_now]['end_ness']
                elif word[-4:] == 'less':
                    emit_p = emit_prob[tag_now]['end_less']
                elif word[-3:] == 'ity':
                    emit_p = emit_prob[tag_now]['end_ity']
                elif word[-1:] == 's':
                    emit_p = emit_prob[tag_now]['end_s']
                # 前缀开始
                elif word[:1] == 'a':
                    emit_p = emit_prob[tag_now]['start_a']
                elif word[:2] == 'be':
                    emit_p = emit_prob[tag_now]['start_be']
                elif word[:2] == 'de':
                    emit_p = emit_prob[tag_now]['start_de']
                elif word[:3] == 'dis':
                    emit_p = emit_prob[tag_now]['start_dis']
                elif word[:2] == 'ex':
                    emit_p = emit_prob[tag_now]['start_ex']
                elif word[:2] == 'in':
                    emit_p = emit_prob[tag_now]['start_in']
                elif word[:3] == 'mis':
                    emit_p = emit_prob[tag_now]['start_mis']
                elif word[:3] == 'non':
                    emit_p = emit_prob[tag_now]['start_non']
                elif word[:4] == 'over':
                    emit_p = emit_prob[tag_now]['start_over']
                elif word[:3] == 'pre':
                    emit_p = emit_prob[tag_now]['start_pre']
                elif word[:2] == 're':
                    emit_p = emit_prob[tag_now]['start_re']
                elif word[:3] == 'uni':
                    emit_p = emit_prob[tag_now]['start_uni']
                elif word[:4] == 'with':
                    emit_p = emit_prob[tag_now]['start_with']
                else:
                    emit_p = emit_prob[tag_now]['UNK']

            log_prob[tag_now] = prev_prob[tag_now] + math.log(emit_p)
            predict_tag_seq[tag_now] = [tag_now]
        return log_prob, predict_tag_seq

    for tag_now in prev_prob.keys():
        v_max = 0
        tag_pre_max = ''
        for tag_pre, prob_pre in prev_prob.items():

            # 超长警告！4
            if word in emit_prob[tag_now]:
                emit_p = emit_prob[tag_now][word]
            else:
                if word[-3:] == 'est':
                    emit_p = emit_prob[tag_now]['end_est']
                elif word[-3:] == 'ion':
                    emit_p = emit_prob[tag_now]['end_ion']
                elif word[-3:] == 'ful':
                    emit_p = emit_prob[tag_now]['end_ful']
                elif word[-2:] == 'ed':
                    emit_p = emit_prob[tag_now]['end_ed']
                elif word[-2:] == 'ly':
                    emit_p = emit_prob[tag_now]['end_ly']
                elif word[-2:] == 'er':
                    emit_p = emit_prob[tag_now]['end_er']
                elif word[-3:] == 'ant':
                    emit_p = emit_prob[tag_now]['end_ant']
                elif word[-2:] == 'th':
                    emit_p = emit_prob[tag_now]['end_th']
                elif word[-4:] == 'ship':
                    emit_p = emit_prob[tag_now]['end_ship']
                elif word[-3:] == 'ing':
                    emit_p = emit_prob[tag_now]['end_ing']
                elif word[-4:] == 'ment':
                    emit_p = emit_prob[tag_now]['end_ment']
                elif word[-4:] == 'ness':
                    emit_p = emit_prob[tag_now]['end_ness']
                # 后缀补充
                elif word[-4:] == 'able':
                    emit_p = emit_prob[tag_now]['end_able']
                elif word[-3:] == 'ise':
                    emit_p = emit_prob[tag_now]['end_ise']
                elif word[-2:] == 'en':
                    emit_p = emit_prob[tag_now]['end_en']
                elif word[-3:] == 'ive':
                    emit_p = emit_prob[tag_now]['end_ive']
                elif word[-2:] == 'or':
                    emit_p = emit_prob[tag_now]['end_or']
                elif word[-4:] == 'ness':
                    emit_p = emit_prob[tag_now]['end_ness']
                elif word[-4:] == 'less':
                    emit_p = emit_prob[tag_now]['end_less']
                elif word[-3:] == 'ity':
                    emit_p = emit_prob[tag_now]['end_ity']
                elif word[-1:] == 's':
                    emit_p = emit_prob[tag_now]['end_s']
                # 前缀开始
                elif word[:1] == 'a':
                    emit_p = emit_prob[tag_now]['start_a']
                elif word[:2] == 'be':
                    emit_p = emit_prob[tag_now]['start_be']
                elif word[:2] == 'de':
                    emit_p = emit_prob[tag_now]['start_de']
                elif word[:3] == 'dis':
                    emit_p = emit_prob[tag_now]['start_dis']
                elif word[:2] == 'ex':
                    emit_p = emit_prob[tag_now]['start_ex']
                elif word[:2] == 'in':
                    emit_p = emit_prob[tag_now]['start_in']
                elif word[:3] == 'mis':
                    emit_p = emit_prob[tag_now]['start_mis']
                elif word[:3] == 'non':
                    emit_p = emit_prob[tag_now]['start_non']
                elif word[:4] == 'over':
                    emit_p = emit_prob[tag_now]['start_over']
                elif word[:3] == 'pre':
                    emit_p = emit_prob[tag_now]['start_pre']
                elif word[:2] == 're':
                    emit_p = emit_prob[tag_now]['start_re']
                elif word[:3] == 'uni':
                    emit_p = emit_prob[tag_now]['start_uni']
                elif word[:4] == 'with':
                    emit_p = emit_prob[tag_now]['start_with']
                else:
                    emit_p = emit_prob[tag_now]['UNK']

            v = prob_pre+math.log(emit_p)+math.log(
                trans_prob[tag_pre].get(tag_now, trans_prob[tag_pre]['UNK']))
            if v > v_max or v_max == 0:
                v_max = v
                tag_pre_max = tag_pre
        log_prob[tag_now] = v_max
        predict_tag_seq[tag_now] = prev_predict_tag_seq[tag_pre_max] + \
            [tag_now]
    return log_prob, predict_tag_seq


def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)

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

        # TODO:(III)
        # according to the storage of probabilities and sequences, get the final prediction.

        max_tag = max(log_prob, key=log_prob.get)
        predicts.append([(sentence[i], predict_tag_seq[max_tag][i])
                         for i in range(length)])
    return predicts
