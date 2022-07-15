# Semantics-for-NLP-Project

##Vader classifier
- preprocess.py holds the preprocessing as described in the manusscript.
- vader_classifier.py computes the results presented in Section 3.1.
- en_tweets_synonyms_fasttext.csv, de_tweets_synonyms_fasttext.csv, en_tweets_synonyms_fasttext.csv holds the embedddings as described in the text.
- tweet_samp_060522_annotate_all.csv holds the raw data with the hand-labeled annotations.
## LSTM/GRU Models
Both the LSTM and GRU models are defined via 50 cells corresponding to the tokens in individual tweets. The models take use of the twitter specific embeddings. They incorporate dropout regularization before the final layer, which is a linear layer with three ouputs corresponding to the three sentiments: positive, negative and neutral.

The models are presented in `lstm.ipynb`. The first section of the notebook outlines data processing for the given datasets, then the process for first the LSTM model and then the GRU model are outlined.

