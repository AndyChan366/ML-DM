# encoding=utf-8
import numpy as np
import pandas as pd
import csv
import re
from random import seed
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# encoding the personality type:
encode = {'INTJ': 0, 'INTP': 1, 'INFJ': 2, 'INFP': 3, 'ISTJ': 4, 'ISTP': 5, 'ISFJ': 6, 'ISFP': 7, 'ENTJ': 8,
              'ENTP': 9, 'ENFJ': 10, 'ENFP': 11, 'ESTJ': 12, 'ESTP': 13, 'ESFJ': 14, 'ESFP': 15}
traintestratio = 0.3
trees = []
# hyper-parameter
maxdepth = 10
minsize = 1
numoftrees = 20


def plogp(p):
    if p == 0.0:
        return 0.0
    else:
        return p * np.log2(p)


def label(i):
    keylist = list((filter(lambda k: encode.get(k) == i, encode.keys())))
    strout = ''.join(keylist)
    return strout


def process(data):
    typeinposts = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ',
                   'ESTP', 'ESFP', 'ESTJ', 'ESFJ', 'infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp',
                   'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
    persontype = []   # store personality type
    postsinfo = []    # store posts
    for row in data.iterrows():
        # remove useless information
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        # remove the words in posts which are related to personality type, prevent the classifier from learning nothing.
        for j in range(len(typeinposts)):
            temp = temp.replace(typeinposts[j], "")
        # get personality and posts after preprocessing:
        label = encode[row[1].type]
        persontype.append(label)
        postsinfo.append(temp)
    # write them into a new file:
    with open("pre.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["type", "posts"])
        for k in range(len(postsinfo)):
            writer.writerow([persontype[k], postsinfo[k]])


def read():
    # read the data after preprocessing.
    data = pd.read_csv("pre.csv")
    perlist = data['type']
    postlist = data['posts']
    perlist = np.array(perlist)
    postlist = np.array(postlist)
    return perlist, postlist


def TFIDF(perlist, postlist):
    # according to experiment and after several attempts to determine the parameter:
    word2vec = CountVectorizer(analyzer="word", max_features=2000, max_df=0.8, min_df=0.05)
    # calculate the frequency of words
    print("WAIT...")
    freq = word2vec.fit_transform(postlist)
    # print(freq)
    # calculate tf-idf matrix:
    tftrans = TfidfTransformer()
    tfidf = tftrans.fit_transform(freq).toarray()
    # print tf-idf matrix:
    # print(tfidf, len(tfidf),len(tfidf[0]))
    tfidf = np.mat(tfidf)
    finals = np.c_[tfidf, perlist]
    # print(finals.shape[0], finals.shape[1])
    # save input features(tf-idf matrix) and output label in 'after.csv'(the last column is label)
    np.savetxt('after.csv', finals, delimiter=',')


# load dataset, the last column is label, others are features.
def loaddata():
    dataset = list()
    with open('after.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            dataset[i][j] = float(dataset[i][j])
    return dataset


# train set:test set=7:3(ratio=0.3)
def split_train_test(dataset, ratio):
    num = len(dataset)
    train_num = int((1-ratio) * num)
    dataset2 = list(dataset)
    traindata = list()
    while len(traindata) < train_num:
        index = randint(0, len(dataset2)-1)    # random choose train set
        traindata.append(dataset2.pop(index))
    testdata = dataset2                        # dataset pop train set, remains are test set.
    return traindata, testdata


def tobinary(data, K):
    # (data, K)==(dataset, the label of positive samples)
    # transfer multi-class into binary-class, convert the label of positive samples into 1, others into 0:
    for i in range(len(data)):
        if data[i][-1] == K:
            data[i][-1] = 1
        else:
            data[i][-1] = 0
    return data


# split of every decision tree
def splitofeachtree(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    if left or right:
        return left, right
    else:
        return None


# Gini index, the smaller value, the better effect
def GINI(groups, class_values):
    gini = 0.0
    total_size = 0
    for group in groups:
        total_size += len(group)
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        for class_value in class_values:
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (size / float(total_size)) * (proportion * (1.0 - proportion))
    return gini


# this function calculate the entropy of leaf node
# due to the entropy of root is a constant(for the same feature)
# the bigger Information Gain, the better effect.
# so the the smaller value of this function, the better effect.
def EntofLeaf(groups, class_values):
    total_size = 0
    leafent = 0.0
    for group in groups:
        total_size += len(group)
    for group in groups:
        temp = 0.0
        size = len(group)
        if size == 0:
            continue
        for class_value in class_values:
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            temp += (plogp(proportion))
        leafent += (temp * (size / float(total_size)))
    return -leafent


# calculate the opposite number of gain ratio, so the smaller value, the better effect
def gainratio(groups, class_values):
    entofleaf = EntofLeaf(groups, class_values)      # get the entropy of leaf which by splitting
    total_size = 0
    ent = 0.0
    for group in groups:
        total_size += len(group)
    for class_value in class_values:
        count = 0
        for group in groups:
            count += [row[-1] for row in group].count(class_value)
        prob = count / float(total_size)
        ent += (plogp(prob))
    ent = -ent                # get the entropy of root
    InfoGain = ent - entofleaf      # calculate the information gain
    entofclass = 0.0
    for group in groups:
        size = len(group)
        prod = size / float(total_size)
        entofclass += (plogp(prod))       # get the opposite number of IV(a) (the denominator of gain ratio)
    ratio = 999.0 if entofclass == 0.0 else (InfoGain / entofclass)    # get the opposite number of gain ratio
    return ratio


# get the best feature to split(use Gini, gain ratio and information gain three criteria)
def bestsplit(dataset, s_features):
    class_values = list(set(row[-1] for row in dataset))  # sort && remove the same
    # print(class_values)
    b_index, b_value, b_score, b_groups = 9999, 9999, 9999, None
    features = list()
    while len(features) < s_features:
        index = randint(0, len(dataset[0]) - 2)  # choose features random
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:             # use values in this feature of each sample to split
            groups = splitofeachtree(index, row[index], dataset)  # divide data into two group
            # as said before(in function define), for these three criteria, the smaller value, the better effect
            # gini = GINI(groups, class_values)         # calculate criteria to judge the best feature for splitting
            # entofleaf = EntofLeaf(groups, class_values)
            gain_ratio = gainratio(groups, class_values)
            if gain_ratio < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gain_ratio, groups
            # if entofleaf < b_score:
            #     b_index, b_value, b_score, b_groups = index, row[index], entofleaf, groups
            # if gini < b_score:
            #     b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# vote for each group
def vote(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# create a sub classifier, recursive classification, until the end of the classification
def subclassifier(node, maxdepth, minsize, s_features, depth):
    left, right = node['groups']
    # print(len(left),len(right))
    del (node['groups'])
    if not left or not right:        # if not split
        node['left'] = node['right'] = vote(left + right)
        return
    # If the classification is not finished yet, choose the result with more tags
    # end the classification ahead of time and prevent over-fitting
    if depth >= maxdepth or len(left) <= minsize or len(right) <= minsize:
        node['left'], node['right'] = vote(left), vote(right)
        return
    else:
        node['left'] = bestsplit(left, s_features)
        subclassifier(node['left'], maxdepth, minsize, s_features, depth + 1)
        node['right'] = bestsplit(right, s_features)
        subclassifier(node['right'], maxdepth, minsize, s_features, depth + 1)


def build_one_tree(train, maxdepth, minsize, s_features):
    root = bestsplit(train, s_features)
    if root['groups'] is None:
        return None
    else:
        subclassifier(root, maxdepth, minsize, s_features, 1)
    return root


# predict with recursion in a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# choose sample random, the size of sample is the square root of data in all.
def choosesample(dataset):
    sample = []
    s_samples = np.sqrt(len(dataset))
    while len(sample) < s_samples:
        index = randint(0, len(dataset) - 2)
        sample.append(dataset[index])
    return sample


# Random Forest Algorithm
def RF(train):
    s_features = np.sqrt(len(train[0]))  # the size of features used to split == the square root of features in all
    for i in range(numoftrees):
        sample = choosesample(train)
        tree = build_one_tree(sample, maxdepth, minsize, s_features)
        trees.append(tree)
    return trees


# vote of random forest(bagging)
def bagging(onetestdata):
    predictions = [predict(tree, onetestdata) for tree in trees]
    return max(set(predictions), key=predictions.count)


# calculate accuracy in test set
def getacc(testdata):
    correct = 0
    for i in range(len(testdata)):
        pre = bagging(testdata[i])
        if testdata[i][-1] == pre:
            correct += 1
    return float(correct) / float(len(testdata))


if __name__ == '__main__':
    # data = pd.read_csv('mbti_1.csv')  # read data
    # process(data)  # preprocess the data, the result is stored in "pre.csv"
    # mbti, info = read()  # read the data in "pre.csv"
    # TFIDF(mbti, info)  # TFIDF, get tfidf-matrix, stored in "after.csv"
    for j in range(16):
        dataset = loaddata()  # load "after.csv"
        dataset = tobinary(dataset, j)   # for every label(16 in all), do a binary classification problem.
        traindata, testdata = split_train_test(dataset, traintestratio)  # get train set and test set
        RF(traindata)                # Random Forest Classifier
        acc = getacc(testdata[:-1])  # get accuracy
        print("the accuracy of {} is:{}".format(label(j), acc))
