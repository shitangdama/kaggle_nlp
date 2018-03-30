import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

train = pd.read_csv("./input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./input/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./input/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "lxml").get_text()      
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join(meaningful_words))

label = train['sentiment']
clean_train_reviews = []
num_reviews = train["review"].size


for i in range(0, num_reviews):
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

clean_test_reviews = [] 

for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english')

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = clean_train_reviews + clean_test_reviews
len_train = len(clean_train_reviews)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print('TF-IDF处理结束.')

model_NB = MNB()
model_NB.fit(train_x, label)
MNB(alpha=1.0, class_prior=None, fit_prior=True)

print("多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))

test_predicted = np.array(model_NB.predict(test_x))
print('保存结果...')
output = pd.DataFrame(data={"id":test["id"], "sentiment":test_predicted})
output.to_csv('nb_output.csv', index=False, quoting=3)
print('结束.')
