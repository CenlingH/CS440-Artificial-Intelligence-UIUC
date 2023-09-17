
# EightPuzzle ------------------------------------------------------------------------------------------------

def read_puzzle(filename):
    with open(filename, "r") as file:
        all_grids = []
        for line in file:
            grid = [[]]
            for c in line.strip():
                if len(grid[-1]) == 3:
                    grid.append([])
                intc = int(c)
                if intc == 0:
                    zero_loc = [len(grid)-1, len(grid[-1])]
                grid[-1].append(intc)
            all_grids.append([grid, zero_loc])
        return all_grids

# WordLadder ------------------------------------------------------------------------------------------------


def read_word_ladders():
    with open("data/word_ladder/ladder_problems.txt", "r") as file:
        all_pairs = []
        for line in file:
            words = line.split()  # 默认空格为分割
            if len(words) > 0:
                all_pairs.append(words)
    return all_pairs


with open("data/word_ladder/wiki-100k.txt", "rb") as word_file:
    english_words = set(word.strip().decode("utf-8").lower()
                        for word in word_file if word[0] != "#")


def is_english_word(word):
    # 返回一个小写set，fruits = {'apple', 'banana', 'cherry', 'apple'}
    return word.lower() in english_words


def levenshteinDistance(s1, s2):  # 两个字符串之间的编辑距离
    if len(s1) > len(s2):
        s1, s2 = s2, s1  # 保证s1是短的

    distances = range(len(s1) + 1)  # 是list，my_list = [1, 2, 3, 4, 5]
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
