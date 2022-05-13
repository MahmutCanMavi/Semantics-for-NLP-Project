# %% 
import matplotlib.pyplot as plt
#from matplotlib.pyplot import hist
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('vader_lexicon')

# %%

df=pd.read_csv("tweet_fetch_050422.csv")
df['text']=df['text'].astype(str)
df['lowercase']=df['text'].map(lambda x: x.lower())

# %%

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    #annotate={"hashtag", "allcaps", "elongated", "repeated",
    #    'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags. #IS IT GOOD IDEA TO KEEP? 
    unpack_contractions=False,  # Unpack contractions (can't -> can not). WORD EMBEDDINGS INCLUDE CONTRACTIONS
    spell_correct_elong=False  # spell correction for elongated words. I'M THINKING IT MIGHT CAUSE TROUBLE WHEN SEVERAL LANGUAGES
                                #DONT THINK THERE WILL BE MANY ELONGATED WORDS IN ECONOMIC TWEETS
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    #tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    #dicts=[emoticons]   COMMENTED. LEAVES EMOTICONS AS EMOTICONS
)

df['pota'] = df['lowercase'].map(text_processor.pre_process_doc)



# %%

sid = SentimentIntensityAnalyzer()
vader_scores=df["pota"].map(sid.polarity_scores)
vader_scores=pd.DataFrame(list(vader_scores))

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
print(np.count_nonzero(vader_scores["compound"]<=-.5))
print(np.count_nonzero(vader_scores["compound"]>=.5))


(123+117)/603













# %%
#We must separate english and german tweets in order to tokenize and take away stop words

en_tweets=df[df['lang']=='en']
de_tweets=df[df['lang']=='de']
#not_en_tw=df[~(df['lang']=="en")]
        ##TODO: REVISAR FUNCION

# %%

#Tokenizing
#en_tweets['tokenized']=en_tweets.apply(lambda row: nltk.word_tokenize(row['text'],language='english'),axis=1)
#de_tweets['tokenized']=de_tweets.apply(lambda row: nltk.word_tokenize(row['text'],language='german'),axis=1)
        #https://www.w3resource.com/python-exercises/nltk/nltk-tokenize-exercise-2.php
        #Will it be a problem that it separated words w apostrophes? can't --> can, 't
        #It separates @ from username!

# %%
#Removing stopwords
en_stopwords=stopwords.words('english')
de_stopwords=stopwords.words('german')

en_tweets['tokenized_no_sw']=np.zeros(len(en_tweets["lowercase"]))
de_tweets['tokenized_no_sw']=np.zeros(len(de_tweets["lowercase"]))

en_tweets['tokenized_no_sw']=en_tweets['pota'].map(lambda x:[word for word in x if not word in en_stopwords])
de_tweets['tokenized_no_sw']=de_tweets['pota'].map(lambda x:[word for word in x if not word in de_stopwords])

# %%
##Reading embeddings


with open('crosslingual_EN-DE_german_twitter_100d_weighted_modified.txt',encoding="Latin-1") as de_embeddings:
    de_emb_dict={}

 
    for i, line in enumerate(de_embeddings):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        de_emb_dict[word] = vector
        print(word)
        print(i)
# %%

en_embeddings=open('crosslingual_EN-DE_english_twitter_100d_weighted_modified.txt',encoding="Latin-1")

en_emb_dict={}
i=0

for line in en_embeddings:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    en_emb_dict[word] = vector
    i=i+1
    print(word)
    print(i)

en_embeddings.close()



# %%









