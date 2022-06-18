#%%
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
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

#%%
df = pd.read_pickle("preprocess_vader_no_hashtags.pkl")
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
df_validation["predicted"].iloc[np.where(vader_scores["compound"] < -.5)[0]] = -1 #can change to 0.33
df_validation["predicted"].iloc[np.where(vader_scores["compound"] > .5)[0]] = 1

#%%
# arithmetic mean of our per-class F1-scores:
#df_validation["predicted"] = np.random.choice(np.array([-1,0,1]), size = 311, replace= True)
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.407,0.406
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.477,0.48
np.round(f1_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #.0479,0.482

np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.409 , 0.409
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.476, 0.477
np.round(precision_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.479, 0.482

np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "macro"), decimals=3) #0.405, 0.404
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "weighted"), decimals=3) #0.479, 0.482
np.round(recall_score(df_validation["annotate_sent"], df_validation["predicted"], average = "micro"), decimals=3) #0.479, 0.482


print(vader_scores)

# %%
plt.hist(vader_scores["neg"])
# %%
plt.hist(vader_scores["neu"])
# %%
plt.hist(vader_scores["pos"])
# %%
plt.hist(vader_scores["compound"])
plt.title("Dataset with unpacked hashtags")
plt.xlabel("Compound VADER score")


# %%
################################# LOOP FOR TESTING HYPOTHESIS ###########################################
iter=100
size_sample=200
df_unpacked = pd.read_pickle("preprocess_vader.pkl")
df_no_hashtags = pd.read_pickle("preprocess_vader_no_hashtags.pkl")
index_annotated = np.where(~np.isnan(df_unpacked["annotate_sent"]))[0]

#Setting both validation sets
np.random.seed(1)
test_and_val_size = np.round(len(index_annotated)*0.2)
test_index = np.random.choice(index_annotated, size = test_and_val_size.astype(int)) # test set
not_test_index = set(index_annotated) - set(test_index)
not_test_index = np.array(list(not_test_index))
val_index = np.random.choice(not_test_index, size = test_and_val_size.astype(int)) # validation set
df_validation_unpacked = df_unpacked.iloc[val_index,:]
df_validation_no_hashtags = df_no_hashtags.iloc[val_index,:]


f1_macro_unpacked=np.zeros(iter)
f1_weighted_unpacked=np.zeros(iter)
f1_micro_unpacked=np.zeros(iter)
f1_macro_no_hashtags=np.zeros(iter)
f1_weighted_no_hashtags=np.zeros(iter)
f1_micro_no_hashtags=np.zeros(iter)

prec_macro_unpacked=np.zeros(iter)
prec_weighted_unpacked=np.zeros(iter)
prec_micro_unpacked=np.zeros(iter)
prec_macro_no_hashtags=np.zeros(iter)
prec_weighted_no_hashtags=np.zeros(iter)
prec_micro_no_hashtags=np.zeros(iter)

rec_macro_unpacked=np.zeros(iter)
rec_weighted_unpacked=np.zeros(iter)
rec_micro_unpacked=np.zeros(iter)
rec_macro_no_hashtags=np.zeros(iter)
rec_weighted_no_hashtags=np.zeros(iter)
rec_micro_no_hashtags=np.zeros(iter)


sid = SentimentIntensityAnalyzer()
for i in range(iter):

    ####UNPACKED DATASET
    np.random.seed(i)
    sample=pd.Series(np.random.choice(df_validation_unpacked["tokenized"],size=size_sample,replace=True)) #bootstraping

    #Running VADER
    vader_scores = sample.map(sid.polarity_scores)
    vader_scores=pd.DataFrame(list(vader_scores))
    df_validation_unpacked["predicted"] = 0
    df_validation_unpacked["predicted"].iloc[np.where(vader_scores["compound"] < -.5)[0]] = -1 #can change to 0.33
    df_validation_unpacked["predicted"].iloc[np.where(vader_scores["compound"] > .5)[0]] = 1

    #Statistics
    f1_macro_unpacked[i]=np.round(f1_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "macro"), decimals=3) 
    f1_weighted_unpacked[i]=np.round(f1_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "weighted"), decimals=3)
    f1_micro_unpacked[i]=np.round(f1_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "micro"), decimals=3)

    prec_macro_unpacked[i]=np.round(precision_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "macro"), decimals=3)
    prec_weighted_unpacked[i]=np.round(precision_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "weighted"), decimals=3)
    prec_micro_unpacked[i]=np.round(precision_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "micro"), decimals=3)

    rec_macro_unpacked[i]=np.round(recall_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "macro"), decimals=3)
    rec_weighted_unpacked[i]=np.round(recall_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "weighted"), decimals=3)
    rec_micro_unpacked[i]=np.round(recall_score(df_validation_unpacked["annotate_sent"], df_validation_unpacked["predicted"], average = "micro"), decimals=3)



    #####NO HASHTAGS DATASET
    np.random.seed(i+1)   #Setting "independence" between datasets
    sample=pd.Series(np.random.choice(df_validation_no_hashtags["tokenized"],size=size_sample,replace=True)) #bootstraping

    #Running VADER
    vader_scores = sample.map(sid.polarity_scores)
    vader_scores=pd.DataFrame(list(vader_scores))
    df_validation_no_hashtags["predicted"] = 0
    df_validation_no_hashtags["predicted"].iloc[np.where(vader_scores["compound"] < -.5)[0]] = -1 #can change to 0.33
    df_validation_no_hashtags["predicted"].iloc[np.where(vader_scores["compound"] > .5)[0]] = 1

    #Statistics
    f1_macro_no_hashtags[i]=np.round(f1_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "macro"), decimals=3) 
    f1_weighted_no_hashtags[i]=np.round(f1_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "weighted"), decimals=3)
    f1_micro_no_hashtags[i]=np.round(f1_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "micro"), decimals=3)

    prec_macro_no_hashtags[i]=np.round(precision_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "macro"), decimals=3)
    prec_weighted_no_hashtags[i]=np.round(precision_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "weighted"), decimals=3)
    prec_micro_no_hashtags[i]=np.round(precision_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "micro"), decimals=3)

    rec_macro_no_hashtags[i]=np.round(recall_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "macro"), decimals=3)
    rec_weighted_no_hashtags[i]=np.round(recall_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "weighted"), decimals=3)
    rec_micro_no_hashtags[i]=np.round(recall_score(df_validation_no_hashtags["annotate_sent"], df_validation_no_hashtags["predicted"], average = "micro"), decimals=3)


# %%
t_stat_f1_macro,pvalue_f1_macro=ttest_ind(f1_macro_unpacked, f1_macro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_f1_macro)
t_stat_f1_weighted,pvalue_f1_weighted=ttest_ind(f1_weighted_unpacked, f1_weighted_no_hashtags,alternative="less",equal_var=False)
print(pvalue_f1_weighted)
t_stat_f1_micro,pvalue_f1_micro=ttest_ind(f1_micro_unpacked, f1_micro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_f1_micro)

t_stat_prec_macro,pvalue_prec_macro=ttest_ind(prec_macro_unpacked, prec_macro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_prec_macro)
t_stat_prec_weighted,pvalue_prec_weighted=ttest_ind(prec_weighted_unpacked, prec_weighted_no_hashtags,alternative="less",equal_var=False)
print(pvalue_prec_weighted)
t_stat_prec_micro,pvalue_prec_micro=ttest_ind(prec_micro_unpacked, prec_micro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_prec_micro)

t_stat_rec_macro,pvalue_rec_macro=ttest_ind(rec_macro_unpacked, rec_macro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_rec_macro)
t_stat_rec_weighted,pvalue_rec_weighted=ttest_ind(rec_weighted_unpacked, rec_weighted_no_hashtags,alternative="less",equal_var=False)
print(pvalue_rec_weighted)
t_stat_rec_micro,pvalue_rec_micro=ttest_ind(rec_micro_unpacked, rec_micro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_rec_micro)


