Execute instruction :python RF.py

"pre.csv": include tags and word vector
"after.csv": include tags and tfidf-matrix

In main function:
if __name__ == "__main__":
    # data = pd.read_csv('mbti_1.csv')  # read data
    # process(data)  # preprocess the data, the result is stored in "pre.csv"
    # mbti,info=read() # read the data in "pre.csv"
    # TFIDF(mbti,info)  # TFIDF, get tfidf-matrix, stored in "after.csv"

When first execute, we need to execute all statements;
After the first execution, we get the data after processing and store them in "pre.csv" && "after.csv", so we needn't to execute above statements.


