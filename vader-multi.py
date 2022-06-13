
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_pickle("preprocess_vader.pkl")
index_annotated = np.where(~np.isnan(df["annotate_sent"]))[0]

np.random.seed(1)
test_and_val_size = np.round(len(index_annotated)*0.2)
test_index = np.random.choice(index_annotated, size = test_and_val_size.astype(int)) # test set
not_test_index = set(index_annotated) - set(test_index)
not_test_index = np.array(list(not_test_index))
val_index = np.random.choice(not_test_index, size = test_and_val_size.astype(int)) # validation set
train_index = np.array(list(set(not_test_index) - set(val_index))) # train set

# vader does not need to be trained actually...
sid = SentimentIntensityAnalyzer()
df_validation = df.iloc[val_index,:]
vader_scores = df_validation["tokenized"].map(sid.polarity_scores)
vader_scores=pd.DataFrame(list(vader_scores))

df_validation["predicted"] = 0
df_validation["predicted"].iloc[np.where(vader_scores["compound"] < -.5)[0]] = -1
df_validation["predicted"].iloc[np.where(vader_scores["compound"] > .5)[0]] = 1

# arithmetic mean of our per-class F1-scores:
#df_validation["predicted"] = np.random.choice(np.array([-1,0,1]), size = 311, replace= True)

np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.245
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.428
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.582



np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.478
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.511
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.476


np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.42
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.476
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.476


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
