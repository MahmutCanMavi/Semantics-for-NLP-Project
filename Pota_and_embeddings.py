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

df=pd.read_csv("tweet_samp_060522_annotate.csv")
df['tweet']=df['tweet'].astype(str)
df['lowercase']=df['tweet'].map(lambda x: x.lower())

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

## missing NER, remove punctation 

#for s in en_tweets["id"]:
#    en_tweets['tokenized_no_sw'][s]=[word for word in en_tweets['pota'][s] if not word in en_stopwords]

# %%
##Reading embeddings

# get embeddings from here: https://drive.google.com/drive/folders/1K0WjPKtar0AvODPAf8VNB1n_ZJAI0WjL?usp=sharing

## German embeddings

with open('crosslingual_EN-DE_german_twitter_100d_weighted_modified.txt',encoding="Latin-1") as de_embeddings:
    de_emb_dict={}
    vocab_de = []

    for i, line in enumerate(de_embeddings):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        de_emb_dict[word] = vector
        vocab_de.append(word)
        #print(word)
        #print(i)

# de_emb_dict["altersgeschützt"] 100-d vector

len(vocab_de) != len(np.unique(vocab_de)) # why?

# concatenate tokens and match unique tokens (i.e. vocabulary) with the emb words 
de_tweets['tokenized_no_sw'] = de_tweets['tokenized_no_sw'].apply(lambda x: np.array(x))
de_tweet_vocab = np.concatenate(de_tweets['tokenized_no_sw'].values)
de_tweet_vocab = list(np.unique(de_tweet_vocab))

tokens_in_vocab = []
for word in de_tweet_vocab:
    tokens_in_vocab.append(word in vocab_de)

np.mean(tokens_in_vocab) # 88% of unique tokens from the german tweets seem to be in the german embedding



# %%

## English embeddings

en_embeddings=open('crosslingual_EN-DE_english_twitter_100d_weighted_modified.txt',encoding="Latin-1")

en_emb_dict={}
vocab_en = []
i=0

for line in en_embeddings:
    values = line.split()
    word = values[0]
    vocab_en.append(word)
    vector = np.asarray(values[1:], dtype='float32')
    en_emb_dict[word] = vector
    i=i+1
    print(word)
    print(i)

en_embeddings.close()



# %%


# de_emb_dict["altersgeschützt"] 100-d vector

len(vocab_en) != len(np.unique(vocab_en)) # why?

# concatenate tokens and match unique tokens (i.e. vocabulary) with the emb words 
en_tweets['tokenized_no_sw'] = en_tweets['tokenized_no_sw'].apply(lambda x: np.array(x))
en_tweet_vocab = np.concatenate(en_tweets['tokenized_no_sw'].values)
en_tweet_vocab = list(np.unique(en_tweet_vocab))

tokens_in_en_vocab = []
tokens_in = []
tokens_not_in = []
for word in en_tweet_vocab:
    is_in = word in vocab_en
    tokens_in_en_vocab.append(is_in)
    if (is_in): tokens_in.append(word)
    if (not is_in): tokens_not_in.append(word)

np.mean(tokens_in_en_vocab) # 87% of unique tokens from the german tweets seem to be in the german embedding
k = np.where(np.array(tokens_in_en_vocab) == False)[0]
en_embeddings.close()

#Another common trick, particularly when working with word embedding based solutions  
#is to replace the word with a nearby word from some form of synonym dictionary. Example : 
#‘I want to know what you are consuming’. Suppose consuming is not in the vocabulary,  replace it with ‘I want to know what you are eating’. 
#Take a look at the following article for more details. 
# https://medium.com/cisco-emerge/creating-semantic-representations-of-out-of-vocabulary-words-for-common-nlp-tasks-842dbdafba18

nltk.download('wordnet')
from nltk.corpus import wordnet
synonym_ready = []

for tok in tokens_not_in:
    synset = wordnet.synsets(tok)
    if (len(synset) == 0): 
        synonym_ready.append("missing")
    else:
        synonym_ready.append(any([sys._name[0:len(sys._name)-5] in vocab_en for sys in synset]))

(unique, counts) = np.unique(synonym_ready, return_counts=True)

# array(['False', 'True', 'missing'], dtype='<U7')
# array([ 53,  64, 624])