from operator import index
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

df = pd.read_pickle("preprocess_vader.pkl")
index_annotated = np.where(~np.isnan(df["annotate_sent"]))[0]

data = df.iloc[index_annotated,:]
df_train, test = train_test_split(data, test_size=0.2, random_state=0)
df_validation, df_test = train_test_split(test, test_size=0.5, random_state=0)
#df_train.to_csv("data/train.csv", index = 0)
#df_validation.to_csv("data/validation.csv", index = 0)
#df_test.to_csv("data/test.csv", index = 0)
 
#np.random.seed(1)
#test_and_val_size = np.round(len(index_annotated)*0.2)
#test_index = np.random.choice(index_annotated, size = test_and_val_size.astype(int)) # test set
#not_test_index = set(index_annotated) - set(test_index)
#not_test_index = np.array(list(not_test_index))
#val_index = np.random.choice(not_test_index, size = test_and_val_size.astype(int)) # validation set
#train_index = np.array(list(set(not_test_index) - set(val_index))) # train set

# vader does not need to be trained actually...
sid = SentimentIntensityAnalyzer()
#df_validation = df.iloc[val_index,:]
vader_scores = df_validation["tokenized"].map(sid.polarity_scores)
vader_scores=pd.DataFrame(list(vader_scores))

df_validation["predicted"] = 0
df_validation["predicted"].iloc[np.where(vader_scores["compound"] < -.5)[0]] = -1
df_validation["predicted"].iloc[np.where(vader_scores["compound"] > .5)[0]] = 1

# arithmetic mean of our per-class F1-scores:
#df_validation["predicted"] = np.random.choice(np.array([-1,0,1]), size = 311, replace= True)
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3)
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) 
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3)

np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3)
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3)
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3)

np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3)
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3)
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3)


print(vader_scores)

# %%
plt.hist(vader_scores["neg"])
# %%
plt.hist(vader_scores["neu"])
# %%
plt.hist(vader_scores["pos"])
# %%
plt.hist(vader_scores["compound"])

# %%
print()
print(np.count_nonzero(vader_scores["compound"]>=.5))


(123+117)/603