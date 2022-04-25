Execute instruction :python svm.py

In main function:
if __name__ == "__main__":
    # data = pd.read_csv('mbti_1.csv')  # read data
    # process(data)  # preprocess the data
    mbti,info=read() # read the data after preprocessing
    arg1,arg2=TFIDF(mbti,info)  # NLP
    SVM(arg1,arg2)  # SVM

When first execute, we need to execute all function:
    data = pd.read_csv('mbti_1.csv')  # read data
    process(data)  # preprocess the data
    mbti,info=read() # read the data after preprocessing
    arg1,arg2=TFIDF(mbti,info)  # NLP
    SVM(arg1,arg2)  # SVM

After the first execution, we get the data after processing and store them in "pre.csv", so we only need to execute following function:
    mbti,info=read() # read the data after preprocessing
    arg1,arg2=TFIDF(mbti,info)  # NLP
    SVM(arg1,arg2)  # SVM

It can reduce the time of process data again. 