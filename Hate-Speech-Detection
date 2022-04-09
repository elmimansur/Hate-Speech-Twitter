# Load relevant packages

import pandas as pd
import numpy as np
import re
import csv
import operator
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None


df = pd.read_csv("./0_data/founta2018_formative2.csv")

random.sample(list(df["tweet"]),10)



def clean_text(text):
            
    # replace mentions and URLs with special token --> provide to students
    text = re.sub(r"@[A-Za-z0-9_-]+",'USR',text)
    text = re.sub(r"http\S+",'URL',text)
    
    # remove newline and tab characters
    text = re.sub("\n", "",text)
    text = re.sub("\t", "",text)
        
    # strip whitespace
    text= text.strip()
    #text = " ".join([w for w in text.split()])
    
    # lowercase
    text = " ".join([w.lower() for w in text.split()])
    
    
    return text

df['tweet'] = df.tweet.map(lambda x: clean_text(x))





full= len(df[df.duplicated(subset=["label","tweet"])])
full

df[df.duplicated()]["label"].value_counts()

dup_tweets= len(df[df.duplicated(subset=["tweet"])])-full
dup_tweets



df=df.drop_duplicates(subset=['label', 'tweet'], keep='last')


binary= {"spam":"non-hateful","normal":"non-hateful","abusive":"non-hateful","hateful":"hateful"}
df["label"]= df.label.map(binary)


df["label"].value_counts()


df["label"].value_counts(normalize=True)



token_pattern = re.compile(r"(?u)\b\w\w+\b")
df["tweet_list"] = df.tweet.apply(lambda x: token_pattern.findall(x))



categories = ["hateful", "non-hateful"]

train_tweets, dev_tweets, test_tweets = dict(), dict(), dict()
X_train, X_test, y_train, y_test = train_test_split(df["tweet_list"], df["label"], test_size=0.15, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(15/85), random_state=123) 



for c_i in categories:
    train_tweets[c_i]=[X_train[i] for i in X_train.index if (y_train[i]==c_i)]
    dev_tweets[c_i]=[X_val[i] for i in X_val.index if (y_val[i]==c_i)]
    test_tweets[c_i]=[X_test[i] for i in X_test.index if (y_test[i]==c_i)]  
    

def all_dict(sample):
    word_list = [w for r in sample for w in r]#.strip().split()]
    # Create a dictionary of words and corresponding frequency counts using a Counter
    vocab_dict = Counter(word_list)
    # Return filtered dictionary
    return Counter({w: c for w, c in vocab_dict.items()})

def dis_dict(sample):
    # Create a list of all words in the reviews
    word_list = [w for r in sample for w in set(r)]#.strip().split())]
    # Create a dictionary of words and corresponding frequency counts using a Counter
    vocab_dict = Counter(word_list)
    # Return filtered dictionary
    return Counter({w: c for w, c in vocab_dict.items()})
    
    
vocab_dicts = dict()
distinct_dicts = dict()

for c_i in categories:
    vocab_dicts[c_i] = all_dict(train_tweets[c_i])
    distinct_dicts[c_i]= dis_dict(train_tweets[c_i])
    
print(vocab_dicts["hateful"].most_common(10))
print(vocab_dicts["non-hateful"].most_common(10))


stop=[]
with open("./0_data/stopwords_formative2.txt","r", newline = '') as f:                                                                                          
    reader = csv.reader(f, delimiter='\n')
    for [i] in reader:
        stop.append(i)
        
print(stop)  


def delete_stop(dicti):
    for c in categories:
        for k in list(dicti[c].keys()):
            if k in stop:
                dicti[c].pop(k)
                

delete_stop(vocab_dicts) 
delete_stop(distinct_dicts)  


wordc={}
wordc["hateful"]=np.sum(list(vocab_dicts["hateful"].values()))
wordc["non-hateful"]=np.sum(list(vocab_dicts["non-hateful"].values()))

print(wordc)


# Naive Bayes without smoothing
# Define function to get P(w|c_i), class-conditional probabilities for w

common_w= list(set(vocab_dicts["hateful"]) & set(vocab_dicts["non-hateful"]))

def delete_dis(dicti):
    for c in categories:
        for k in list(dicti[c].keys()):
            if k not in common_w:
                dicti[c].pop(k)
                
delete_stop(vocab_dicts)
delete_stop(distinct_dicts)

import math

vocab_dicts["non-hateful"]

(vocab_dicts["non-hateful"]["imagine"])/(np.sum(list(vocab_dicts["non-hateful"].values())))

len(vocab_dicts.items())

for l,word in vocab_dicts.items():
    print(l)

def Pwc(diction=vocab_dicts):
    pw={}
    for l,word in vocab_dicts.items():
        pw[l]={}
        for k,f in word.items():
            pr= f/wordc[l]
            #pr=(vocab_dicts[clas][word])/wordc[clas] #Freq of word/Total words (in class)= Probability of word in a class
            pw[l][k]=np.log(pr) #np.log(pr)
    return pw 
    
labels.count("non-hateful")

def Pc(labels):
    a= labels.count("hateful")
    b= labels.count("non-hateful")
    
    prob_clas={}
    prob_clas["hateful"]=np.log(a/(a+b))
    prob_clas["non-hateful"]=np.log(b/(a+b))
    return prob_clas
    

Pc()

def pred1(t_set=train_tweets):
    tweets=[f for k,v in train_tweets.items() for f in train_tweets[k]]
    labels=[k for k,v in train_tweets.items() for f in train_tweets[k]]
    
    pred= []
    
    prob_class= Pc(labels)
    pw= Pwc()
    for i in range(len(tweets)):
        scores={"hateful":0,"non-hateful":0}
        for word in tweets[i]:
            if word in common_w:
                scores["hateful"]+=pw["hateful"][word]
                scores["non-hateful"]+=pw["non-hateful"][word]
        
        scores["hateful"]+= prob_class["hateful"]
        scores["non-hateful"]+= prob_class["non-hateful"]
        
        cl=max(scores, key=scores.get)
        pred.append((cl,1 if tweets[i] in train_tweets[cl] else 0))
        
        
    return pred

def pred1(t_set=train_tweets):
    tweets=[f for k,v in train_tweets.items() for f in train_tweets[k]]
    labels=[k for k,v in train_tweets.items() for f in train_tweets[k]]
    
    pred= []
    
    prob_class= Pc(labels)
    pw= Pwc()
    for i in range(len(tweets)):
        scores={"hateful":0,"non-hateful":0}
        for word in tweets[i]:
            if word in common_w:
                scores["hateful"]+=pw["hateful"][word]
                scores["non-hateful"]+=pw["non-hateful"][word]
        
        scores["hateful"]+= prob_class["hateful"]
        scores["non-hateful"]+= prob_class["non-hateful"]
        
        cl=max(scores, key=scores.get)
        pred.append((cl,cl==labels[i]))
        
        
    return pred

pred=pred1()

summ=0
for i in pred:
    (h,[v])=i
    summ+=v
summ/len(pred)
