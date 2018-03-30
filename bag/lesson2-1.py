from gensim.models import word2vec
import gensim

model = word2vec.Word2Vec.load("./300features_40minwords_10context")
print(model.doesnt_match("man woman child kitchen".split()))