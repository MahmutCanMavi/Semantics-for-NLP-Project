# Semantics-for-NLP-Project
## LSTM/GRU Models
Both the LSTM and GRU models are defined via 50 cells corresponding to the tokens in individual tweets. The models take use of the twitter specific embeddings. They incorporate dropout regularization before the final layer, which is a linear layer with three ouputs corresponding to the three sentiments: positive, negative and neutral.

The models are presented in `lstm.ipynb`. The first section of the notebook outlines data processing for the given datasets, then the process for first the LSTM model and then the GRU model are outlined.