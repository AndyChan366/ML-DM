import numpy as np
import pandas as pd
import csv
import re
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


def process(data):
    # encoding the personality type:
    encode = {'INTJ': 0, 'INTP': 1, 'INFJ': 2, 'INFP': 3, 'ISTJ': 4, 'ISTP': 5, 'ISFJ': 6, 'ISFP': 7, 'ENTJ': 8, 'ENTP': 9, 'ENFJ': 10, 'ENFP': 11, 'ESTJ': 12, 'ESTP': 13, 'ESFJ': 14, 'ESFP': 15}
    typeinposts = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ', 'infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
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
    # read the data after preprocessing, this function can reduce the call of process(), so it save time.
    data = pd.read_csv("pre.csv")
    perlist = data['type']
    postlist = data['posts']
    perlist = np.array(perlist)
    postlist = np.array(postlist)
    return perlist, postlist

def TFIDF(perlist,postlist):
    # according to experiment and after several attempts to determine the parameter:
    word2vec = CountVectorizer(analyzer="word",max_features=2000,max_df=0.8,min_df=0.05)
    # calculate the frequency of words
    print("WAIT...")
    freq = word2vec.fit_transform(postlist)
    # print(freq)
    # calculate tf-idf matrix:
    tftrans = TfidfTransformer()
    tfidf = tftrans.fit_transform(freq).toarray()
    # print tf-idf matrix:
    # print(tfidf, len(tfidf),len(tfidf[0]))
    return tfidf, perlist

def tobinary(train, test, K):
    # (train, test, K)==(y_train, y_test, the label of positive samples)
    # transfer muliti-class into binary-class, convert the label of positive samples into 1, the label of other 15 types into -1:
    temp1 = np.zeros(len(train))
    for i in range(len(train)):
        temp1[i] = 1 if train[i]==K else -1
    train_Y = np.array(temp1, dtype=np.int)
    temp2 = np.zeros(len(test))
    for i in range(len(test)):
        temp2[i] = 1 if test[i]==K else -1
    test_Y = np.array(temp2, dtype=np.int)
    return train_Y, test_Y

def SVM(tfidf, perlist):
    typeclass = ['INTJ','INTP','INFJ','INFP','ISTJ','ISTP','ISFJ','ISFP','ENTJ','ENTP','ENFJ','ENFP','ESTJ','ESTP','ESFJ','ESFP']
    X = tfidf
    Y = perlist
    randomseed = 42
    size = 0.3
    # split data into train set and test set with proportion 7:3(according to experiment):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=randomseed)
    kernels = ["linear","rbf","poly","sigmoid"]
    # kernel = "linear"
    # kernel = "rbf"
    # kernel = "poly"
    # kernel = "sigmoid"
    # code for cross validation(K-Fold, where K=10):
    # Ctemp = [0.5, 1.0, 2.0]     # hyper-parameter, determined by cross validation.
    # for C in Ctemp:
    #     for m in range(len(typeclass)):
    #         train_Y, test_Y = tobinary(Y_train, Y_test, m)
    #         # for kernel in kernels:
    #         method = svm.SVC(decision_function_shape='ovr', kernel=kernel, C=C)
    #         model = method.fit(X_train, train_Y)
    #         scores = cross_val_score(method, X_train, train_Y, cv=10, scoring='accuracy')
    #         print("C={},score:{}".format(C, scores.mean()))
    # predict the result in test set:
    for m in range(len(typeclass)):
        train_Y, test_Y = tobinary(Y_train, Y_test, m)
        for kernel in kernels:
            method = svm.SVC(decision_function_shape='ovr', kernel=kernel, C=1)
            model = method.fit(X_train, train_Y)
            output = model.predict(X_test)
            accuracy = accuracy_score(test_Y, output)
            print("{} for {} Accuracy:{}%".format(typeclass[m], kernel, accuracy * 100.0))


if __name__ == "__main__":
    # data = pd.read_csv('mbti_1.csv')  # read data
    # process(data)  # preprocess the data
    mbti,info=read() # read the data after preprocessing
    arg1,arg2=TFIDF(mbti,info)  # NLP
    SVM(arg1,arg2)  # SVM
