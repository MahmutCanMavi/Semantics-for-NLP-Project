import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

df = pd.read_pickle("preprocess_vader.pkl")

sid = SentimentIntensityAnalyzer()
vader_scores=df["tokenized"].map(sid.polarity_scores) # misnomer, not actually tokenized
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