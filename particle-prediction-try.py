import re
import numpy as np
import pandas as pd
import torch
import nltk

from tqdm import tqdm
from collections import defaultdict, Counter
from html import unescape
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F

pd.options.mode.chained_assignment = None

nltk.download('punkt')

# Define function to clean and split text
def clean(text):
    text = unescape(text)
    return [re.sub('[^a-z0-9]', '', w.lower()) for w in text.strip().split()]

# Load dataframe
df = pd.read_csv('reddit_mbti.csv')

# Remove columns not needed for formative
df = df[['comment', 'type']]

# Remove empty comments
df = df[df.comment.apply(lambda x: len(clean(x))) > 0]

# Define dictionary for particle look-up
p2id = {'over': 0, 'back': 1, 'around': 2, 'out': 3}

# Define dictionary for reverse particle look-up
id2p = {v: k for k, v in p2id.items()}

# Initialize lists for storing contexts around particles
sent_1 = list()
sent_2 = list()

# Initialize list for storing labels
labels = list()

# Loop over comments
for c in tqdm(df.comment):
    
    # Loop over individual sentences
    for s in nltk.sent_tokenize(c):
        
        # Clean and split sentence
        split = clean(s)
        
        if len(split) < 10:
            continue
        
        # Add sentence to list if only one particle in sentence
        if len([w for w in split if w in p2id]) == 1:
            
            # Identify particle
            p = [w for w in split if w in p2id][0]

            # Store contexts and label
            sent_1.append(split[:split.index(p)])
            sent_2.append(split[split.index(p) + 1:])
            labels.append(p)

# Create dataframe with contexts and labels and perform stratified sampling
p_df = pd.DataFrame({'sent_1': sent_1, 'sent_2': sent_2, 'label': labels})[['sent_1', 'sent_2', 'label']]
p_df = p_df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=1500, random_state=123)).reset_index(drop=True)

# Split dataframe into training, evaluation, and test data
train, dev_test = train_test_split(p_df, test_size=0.2, stratify=p_df['label'], random_state=123)
dev, test = train_test_split(dev_test, test_size=0.5, stratify=dev_test['label'], random_state=123)
