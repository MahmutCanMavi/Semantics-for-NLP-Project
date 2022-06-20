
#%%

####NOTE: For this script you need to install vader-multi
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


df = pd.read_pickle("preprocess_vader_no_hashtags.pkl")
index_annotated = np.where(~np.isnan(df["annotate_sent"]))[0]

np.random.seed(1)
test_and_val_size = np.round(len(index_annotated)*0.2)
test_index = np.random.choice(index_annotated, size = test_and_val_size.astype(int)) # test set
not_test_index = set(index_annotated) - set(test_index)
not_test_index = np.array(list(not_test_index))
val_index = np.random.choice(not_test_index, size = test_and_val_size.astype(int)) # validation set
train_index = np.array(list(set(not_test_index) - set(val_index))) # train set


sid = SentimentIntensityAnalyzer()
df_validation = df.iloc[val_index,:]

#vader-multi needs indexes to match row number
real_index=df_validation["tokenized"].index
df_validation.index=range(df_validation.shape[0])

vader_scores=np.zeros(len(df_validation["tokenized"]))
for tok in range(len(df_validation["tokenized"])) :
    print(sid.polarity_scores(df_validation["tokenized"][tok]))

#Setting indexes back to original value
df_validation.index=real_index
vader_scores=pd.DataFrame(list(vader_scores))

df_validation["predicted"] = 0
df_validation["predicted"].iloc[np.where(vader_scores["compound"] < -.5)[0]] = -1
df_validation["predicted"].iloc[np.where(vader_scores["compound"] > .5)[0]] = 1

# arithmetic mean of our per-class F1-scores:
#df_validation["predicted"] = np.random.choice(np.array([-1,0,1]), size = 311, replace= True)

np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.407
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.477
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.479



np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.409
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.476
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.479


np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.405
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.479
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.479


#%%

print(vader_scores)

# %%
plt.hist(vader_scores["neg"])
# %%
plt.hist(vader_scores["neu"])
# %%
plt.hist(vader_scores["pos"])
# %%
plt.hist(vader_scores["compound"])
