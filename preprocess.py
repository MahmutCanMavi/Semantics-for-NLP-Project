import pandas as pd
import numpy as np
import nltk  # nltk-3.7
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.corpus import wordnet
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ner import remove_entities # see ner.py

## Merge annotated data set (1500 tweets) with the full one-year sample (8761 tweets)

annotated_tweets = pd.read_csv("tweet_samp_060522_annotate_all.csv")
annotated_tweets = annotated_tweets.drop("Unnamed: 0", axis=1)
annotated_tweets = annotated_tweets.drop("Unnamed: 0.1", axis=1)

all_tweets = pd.read_csv("tweet_samp_060522.csv")
all_tweets = all_tweets.drop("Unnamed: 0", axis=1)

cols = all_tweets.columns.tolist()
d = pd.merge(all_tweets, annotated_tweets,  how='outer', on=cols)
d = d[d['lang'] != 'tl'] # one entry has wrong langauge
d.index=range(d.shape[0])

d['tweet'] = d['tweet'].astype('str') # maybe try astype('unicode') and see if out-of-vocab/performance is affected
d['lowercase'] = d['tweet'].map(lambda x: x.lower()) # is NER affected by capitalization?
#d = d.iloc[np.random.choice(range(0, d.shape[0] + 1), size = 500),:] # quick test runs

## Does Baziotis (2017) coincide with Pota (2020)?
# https://github.com/cbaziotis/ekphrasis
# install via pip install git+https://github.com/fucaja/ekphrasis.git (uses different fork which does not requier url download for word stats)

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
    spell_correct_elong=False,  # spell correction for elongated words. I'M THINKING IT MIGHT CAUSE TROUBLE WHEN SEVERAL LANGUAGES
                                #DONT THINK THERE WILL BE MANY ELONGATED WORDS IN ECONOMIC TWEETS
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    #dicts=[emoticons]   COMMENTED. LEAVES EMOTICONS AS EMOTICONS
)

d['tokenized'] = d['lowercase'].map(text_processor.pre_process_doc)

# For vader classifier comment out "tokenizer=SocialTokenizer()"
#d.to_pickle("preprocess_vader.pkl")

# separate english and german tweets
en_tweets = d[d['lang']=='en'] # 6858/8760 = 0.78
de_tweets = d[d['lang']=='de'] # 1902/8760 = 0.22

### remove stopwords
en_stopwords = stopwords.words('english')
de_stopwords = stopwords.words('german')

# english
en_tweets['tokenized_no_sw']=np.zeros(len(en_tweets["tokenized"]))
en_tweets['tokenized_no_sw']=en_tweets['tokenized'].map(lambda x:[word for word in x if not word in en_stopwords])

# german
de_tweets['tokenized_no_sw']=np.zeros(len(de_tweets["tokenized"]))
de_tweets['tokenized_no_sw']=de_tweets['tokenized'].map(lambda x:[word for word in x if not word in de_stopwords])

### apply named entity recognition

# already done with commented code, use converters to get list instead of string for pandas column
de_tweets = pd.read_csv("de_tweets.csv", converters={'preproc': pd.eval, 'tokenized_no_sw': pd.eval})
en_tweets = pd.read_csv("en_tweets.csv", converters={'preproc': pd.eval, 'tokenized_no_sw': pd.eval})

# after ner: some tweets are empty for some reason...
de_tweets["preproc"].iloc[511] = de_tweets["tokenized_no_sw"].iloc[511]
en_tweets["preproc"].iloc[1959] = en_tweets["tokenized_no_sw"].iloc[1959]
en_tweets["preproc"].iloc[2035] = en_tweets["tokenized_no_sw"].iloc[2035]
en_tweets["preproc"].iloc[2041] = en_tweets["tokenized_no_sw"].iloc[2041]

## DISCUSS: replace the following token by some vector that is available and matches the content
for idx in range(de_tweets['preproc'].shape[0]):

    de_tweets['preproc'].iloc[idx] = [tok.replace('<user>','paula') for tok in de_tweets['preproc'].iloc[idx]]
    de_tweets['preproc'].iloc[idx] = [tok.replace('<percent>','anteil') for tok in de_tweets['preproc'].iloc[idx]]
    de_tweets['preproc'].iloc[idx] = [tok.replace('<number>','192') for tok in de_tweets['preproc'].iloc[idx]]
    de_tweets['preproc'].iloc[idx] = [tok.replace('<money>','geld') for tok in de_tweets['preproc'].iloc[idx]]
    de_tweets['preproc'].iloc[idx] = [tok.replace('<date>','september') for tok in de_tweets['preproc'].iloc[idx]]
    de_tweets['preproc'].iloc[idx] = [tok.replace('<time>','10:16') for tok in de_tweets['preproc'].iloc[idx]]

for idx in range(en_tweets['preproc'].shape[0]):

    en_tweets['preproc'].iloc[idx] = [tok.replace('<user>','paula') for tok in en_tweets['preproc'].iloc[idx]]
    en_tweets['preproc'].iloc[idx] = [tok.replace('<percent>','fraction') for tok in en_tweets['preproc'].iloc[idx]]
    en_tweets['preproc'].iloc[idx] = [tok.replace('<number>','192') for tok in en_tweets['preproc'].iloc[idx]]
    en_tweets['preproc'].iloc[idx] = [tok.replace('<money>','money') for tok in en_tweets['preproc'].iloc[idx]]
    en_tweets['preproc'].iloc[idx] = [tok.replace('<date>','september') for tok in en_tweets['preproc'].iloc[idx]]
    en_tweets['preproc'].iloc[idx] = [tok.replace('<time>','10:16') for tok in en_tweets['preproc'].iloc[idx]]

# to check the effect without NER
#de_tweets["preproc"] = de_tweets["tokenized_no_sw"]
#en_tweets["preproc"] = en_tweets["tokenized_no_sw"]

# ## german
# de_tweets["tokenized_no_sw_and_entities"] = np.zeros(len(de_tweets["tokenized_no_sw"]))

# # Since the indices are differenent for the dataset (result of them being dissections of one df)
# for i in range(0,de_tweets.shape[0]):
#     de_tweets["tokenized_no_sw_and_entities"].iloc[i] = remove_entities(de_tweets["tweet"].iloc[i], de_tweets["tokenized_no_sw"].iloc[i].copy())

# ## english
# en_tweets["tokenized_no_sw_and_entities"] = np.zeros(len(en_tweets["tokenized_no_sw"]))

# for i in range(0,en_tweets.shape[0]):
#     en_tweets["tokenized_no_sw_and_entities"].iloc[i] = remove_entities(en_tweets["tweet"].iloc[i], en_tweets["tokenized_no_sw"].iloc[i].copy())

# # rename for the sake of brevity
# de_tweets = de_tweets.rename(columns={'tokenized_no_sw_and_entities': 'preproc'})
# en_tweets = en_tweets.rename(columns={'tokenized_no_sw_and_entities': 'preproc'})

# since NER takes time...
#de_tweets.to_csv("de_tweets.csv", index=False)
#en_tweets.to_csv("en_tweets.csv", index=False)

### read embeddings
# get embeddings from here: https://drive.google.com/drive/folders/1K0WjPKtar0AvODPAf8VNB1n_ZJAI0WjL?usp=sharing

## german

with open('crosslingual_EN-DE_german_twitter_100d_weighted_modified.txt', encoding="Latin-1") as de_embeddings:
    de_emb_dict={}
    de_emb_vocab = []
    for i, line in enumerate(de_embeddings):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        de_emb_dict[word] = vector
        de_emb_vocab.append(word)

# they should be the same shouldnt they?
#len(de_emb_vocab) == len(np.unique(de_emb_vocab))

# concatenate tokens and match unique tokens from corpus with the emb vocab
de_tweets['preproc'] = de_tweets['preproc'].apply(lambda x: np.array(x))
de_tweet_corp_all = np.concatenate(de_tweets['preproc'].values)
de_tweet_corp = list(np.unique(de_tweet_corp_all))
len(de_tweet_corp_all) # 45123
len(np.unique(de_tweet_corp_all)) # 6837

# check which tokens have a emb vector available 
#np.mean([emb_token in de_emb_vocab for emb_token in de_tweet_corp_all]) # non-unique: 0.77

tokens_in_de_emb = []
tokens_in_de = []
tokens_not_in_de = []
for token in de_tweet_corp:
    is_in = token in de_emb_vocab
    tokens_in_de_emb.append(is_in) #np.mean(tokens_in_de_emb)
    if (is_in): tokens_in_de.append(token)
    if (not is_in): tokens_not_in_de.append(token)

## replace out-of-vocab tokens with synonyms to alleviate oov problem
# https://medium.com/cisco-emerge/creating-semantic-representations-of-out-of-vocabulary-words-for-common-nlp-tasks-842dbdafba18
# German synonym vocab needs a signed licence agreement, not worth the effort
# https://docs.google.com/document/d/1rdn0hOnJNcOBWEZgipdDfSyjJdnv_sinuAUSDSpiQns/edit?hl=en
# https://linguistics.stackexchange.com/questions/15471/germanltk-not-finding-files-python
# maybe the english one has matches, too

de_has_synonyms = []
de_synonyms = {}
for tok in tokens_not_in_de:
    synset = wordnet.synsets(tok) # english wordnet unfortunately
    if (len(synset) == 0): 
        de_has_synonyms.append("missing")
    else:
        for sys in synset:
            synonym = sys._name[0:len(sys._name)-5]
            has_synonym = synonym in de_emb_vocab
            de_has_synonyms.append(has_synonym)
            if (has_synonym): 
                de_synonyms[tok] = synonym
                break # sometimes there are multiple synonyms

(de_unique, de_counts) = np.unique(de_has_synonyms, return_counts = True)

# replace words in de corpus with synonyms
for idx in range(de_tweets['preproc'].shape[0]):
    for key, value in de_synonyms.items():
        true_false = de_tweets['preproc'].iloc[idx] == key
        de_tweets['preproc'].iloc[idx][true_false] = value

tokens_in_de = tokens_in_de + list(de_synonyms.keys())

## english

with open('crosslingual_EN-DE_english_twitter_100d_weighted_modified.txt', encoding="Latin-1") as en_embeddings:
    en_emb_dict={}
    en_emb_vocab = []
    for i, line in enumerate(en_embeddings):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        en_emb_dict[word] = vector
        en_emb_vocab.append(word)

# they should be the same shouldnt they?
#len(en_emb_vocab) == len(np.unique(en_emb_vocab))

# concatenate tokens and match unique tokens from corpus with the emb vocab
en_tweets['preproc'] = en_tweets['preproc'].apply(lambda x: np.array(x))
en_tweet_corp_all = np.concatenate(en_tweets['preproc'].values)
en_tweet_corp = list(np.unique(en_tweet_corp_all))
len(en_tweet_corp_all) # 175315
len(np.unique(en_tweet_corp_all)) # 12830

# check which tokens have a emb vector available 
#np.mean([emb_token in en_emb_vocab for emb_token in en_tweet_corp_all]) # non-unique: 0.78

tokens_in_en_vocab = []
tokens_in_en = []
tokens_not_in_en = []
for token in en_tweet_corp:
    is_in = token in en_emb_vocab
    tokens_in_en_vocab.append(is_in)
    if (is_in): tokens_in_en.append(token)
    if (not is_in): tokens_not_in_en.append(token)

#np.mean(tokens_in_en_vocab)

## replace out-of-vocab tokens with synonyms to alleviate oov problem

en_has_synonyms = []
en_synonyms = {}
for tok in tokens_not_in_en:
    synset = wordnet.synsets(tok) # english wordnet unfortunately
    if (len(synset) == 0): 
        en_has_synonyms.append("missing")
    else:
        for sys in synset:
            synonym = sys._name[0:len(sys._name)-5]
            has_synonym = synonym in en_emb_vocab
            en_has_synonyms.append(has_synonym)
            if (has_synonym): 
                en_synonyms[tok] = synonym
                break # sometimes there are multiple synonyms

(en_unique, en_counts) = np.unique(en_has_synonyms, return_counts = True)

# replace words in en corpus with synonyms
for idx in range(en_tweets['preproc'].shape[0]):
    for key, value in en_synonyms.items():
        true_false = en_tweets['preproc'].iloc[idx] == key
        en_tweets['preproc'].iloc[idx][true_false] = value

# although they are not in the main vocab we can substitute them by synonyms and thus count them in
tokens_in_en = tokens_in_en + list(en_synonyms.keys())

## one further way to alleviate oov problem

# noticed that some english words where in the german tweets, match with english vocab
de_tok = [tok in en_emb_vocab for tok in tokens_not_in_de]
add_from_en = list(np.asarray(en_emb_vocab, dtype=object)[np.where(de_tok)[0]])
tokens_in_de = tokens_in_de + add_from_en
len(tokens_in_de)/len(de_tweet_corp)
de_oov = set(de_tweet_corp) - set(tokens_in_de) # german oov words
np.mean([emb_token in tokens_in_de for emb_token in de_tweet_corp_all]) 

# frequency of how many times each token is missing in the corpus
abs_freq = {}
for tok in de_oov:
    abs_freq[tok] = np.sum(de_tweet_corp_all == tok)
abs_freq = sorted(abs_freq.items(), key=lambda x: x[1])

# noticed that some german words where in the english tweets, match with german vocab
en_tok = [tok in de_emb_vocab for tok in tokens_not_in_en]
add_from_de = list(np.asarray(de_emb_vocab, dtype=object)[np.where(en_tok)[0]])
tokens_in_en = tokens_in_en + add_from_de
len(tokens_in_en)/len(en_tweet_corp)
en_oov = set(en_tweet_corp) - set(tokens_in_en) # english oov words
np.mean([emb_token in tokens_in_en for emb_token in en_tweet_corp_all]) 

# frequency of how many times each token is missing in the corpus
abs_freq = {}
for tok in en_oov:
    abs_freq[tok] = np.sum(en_tweet_corp_all == tok)
abs_freq = sorted(abs_freq.items(), key=lambda x: x[1])
