# Attention and external Knowledge augmented Dual Encoder with bi-directional GRU (AK-DE-biGRU)

Code for implementing the paper : "Improving Response Selection in Multi-turn Dialogue Systems by Incorporating Domain Knowledge" 

## Getting Started

We use python version 3.6.4
Install the requirements.txt file and install pytorch version: "0.3.1.post2"

### Prerequisites

Download the pre-processed files from Wu et. al, from here: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu%20data.zip?dl=0and save it in ubuntu_data.
Run: python ./preprocess.py
To create the required preprocessed dataset
This will be read from data.py
Use the train.txt file to train a fasttext model using the fasttext library:https://github.com/facebookresearch/fastText by:
./fasttext skipgram -input train.txt -dim 200 -output fast_text_200

Save this file into a numpy array whose index corresponds to the word_id from the previous dictionary and the row contains the fasttext vector for that word.
copy the file to ubuntu_data directory.

Download the ubuntu_description.npy file provided and copy it to ubuntu_data directory

## Running the model

The AK-DKE-biGRU model should be run as:
```
python -u run_models.py --h_dim 300 --mb_size 32 --n_epoch 20 --gpu --lr 0.0001
```
