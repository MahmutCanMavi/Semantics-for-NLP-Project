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
from sklearn.model_selection import train_test_split

#%%
#Getting validation indexes, getting them from preprocess_vader/ _no_hashtags
validation = pd.read_csv("data/validation.csv")   #Validation.csv is no_hashtags. 
full_df=pd.read_pickle("preprocess_vader_no_hashtags.pkl")

idx=pd.Series.to_list(validation["tweet_id"])
mask=full_df['tweet_id'].isin(idx)
val_df=full_df.loc[mask]



#Running Vader model and setting 0.5 threshold
sid = SentimentIntensityAnalyzer()
vader_scores = val_df["tokenized"].map(sid.polarity_scores)
vader_scores=pd.DataFrame(list(vader_scores))

val_df["predicted"] = 0
val_df["predicted"].iloc[np.where(vader_scores["compound"] < -.5)[0]] = -1 #can change to 0.33
val_df["predicted"].iloc[np.where(vader_scores["compound"] > .5)[0]] = 1

#%%
# Results for F1, precision and recall.
#Commented results: preprocess_vader_no_hashtags, preprocess_vader

np.round(f1_score(val_df["annotate_sent"], val_df["predicted"], average = "macro"), decimals=3) #0.388,0.397
np.round(f1_score(val_df["annotate_sent"], val_df["predicted"], average = "weighted"), decimals=3) #0.449,0.456
np.round(f1_score(val_df["annotate_sent"], val_df["predicted"], average = "micro"), decimals=3) #.0447,0.453

np.round(precision_score(val_df["annotate_sent"], val_df["predicted"], average = "macro"), decimals=3) #0.488 , 0.396
np.round(precision_score(val_df["annotate_sent"], val_df["predicted"], average = "weighted"), decimals=3) #0.451, 0.459
np.round(precision_score(val_df["annotate_sent"], val_df["predicted"], average = "micro"), decimals=3) #0.447, 0.453

np.round(recall_score(val_df["annotate_sent"], val_df["predicted"], average = "macro"), decimals=3) #0.389, 0.397
np.round(recall_score(val_df["annotate_sent"], val_df["predicted"], average = "weighted"), decimals=3) #0.447, 0.453
np.round(recall_score(val_df["annotate_sent"], val_df["predicted"], average = "micro"), decimals=3) #0.447, 0.453


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
size_sample=311
df_unpacked = pd.read_pickle("preprocess_vader.pkl")
df_no_hashtags = pd.read_pickle("preprocess_vader_no_hashtags.pkl")

mask=df_unpacked['tweet_id'].isin(idx)


df_validation_unpacked=df_unpacked.loc[mask]
df_validation_no_hashtags=df_no_hashtags.loc[mask]


#%% 
iter=100
size_sample=311
df_unpacked = pd.read_pickle("preprocess_vader.pkl")
df_no_hashtags=pd.read_pickle("preprocess_vader_no_hashtags.pkl")

mask1=df_no_hashtags['tweet_id'].isin(idx)
mask2=df_unpacked['tweet_id'].isin(idx)
df_validation_unpacked=df_no_hashtags.loc[mask1]
df_validation_no_hashtags=df_unpacked.loc[mask2]

#%%
'''





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

'''
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
print(pvalue_f1_macro) #0.5301
t_stat_f1_weighted,pvalue_f1_weighted=ttest_ind(f1_weighted_unpacked, f1_weighted_no_hashtags,alternative="less",equal_var=False)
print(pvalue_f1_weighted) #0.6359
t_stat_f1_micro,pvalue_f1_micro=ttest_ind(f1_micro_unpacked, f1_micro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_f1_micro)#0.7149

t_stat_prec_macro,pvalue_prec_macro=ttest_ind(prec_macro_unpacked, prec_macro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_prec_macro) #0.5333
t_stat_prec_weighted,pvalue_prec_weighted=ttest_ind(prec_weighted_unpacked, prec_weighted_no_hashtags,alternative="less",equal_var=False)
print(pvalue_prec_weighted) #0.5215
t_stat_prec_micro,pvalue_prec_micro=ttest_ind(prec_micro_unpacked, prec_micro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_prec_micro) #0.7149

t_stat_rec_macro,pvalue_rec_macro=ttest_ind(rec_macro_unpacked, rec_macro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_rec_macro) #0.5315
t_stat_rec_weighted,pvalue_rec_weighted=ttest_ind(rec_weighted_unpacked, rec_weighted_no_hashtags,alternative="less",equal_var=False)
print(pvalue_rec_weighted) #0.7149
t_stat_rec_micro,pvalue_rec_micro=ttest_ind(rec_micro_unpacked, rec_micro_no_hashtags,alternative="less",equal_var=False)
print(pvalue_rec_micro) #0.7149



# %%
