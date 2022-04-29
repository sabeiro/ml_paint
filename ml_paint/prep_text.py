import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import warnings; warnings.simplefilter('ignore')
import nltk
import string
from nltk import ngrams
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')

baseDir = "/home/sabeiro/tmp/pers/text/"
cName = "anima_raw"
fName = baseDir + cName + ".txt"
textD = pd.read_csv(fName,sep=":",header=None)
text = textD[2]
text = text.str.strip()
text = text.str.replace("\t", "")
text = text.str.replace('"', "")
text = text.str.replace("^ ", "")
text = text.str.replace("<Media omitted>", "")
text = text.str.replace("https", "")
text = text[~text.isna()]
printable = list(set(string.printable)) + ['à','è','ò','ù','é','ó',"ö","ä","ü"]
text = text.apply(lambda s: ''.join(filter(lambda x: x in printable, s)) )
setL = text == ''
text = text[~setL]
text.to_csv(baseDir+"anima"+".txt",index=False,header=False,doublequote=False,quoting=False,escapechar="\\") #,quotechar=""




# text = open(fName, encoding="utf-8").read()
text = text.lower()
text = text.translate(str.maketrans("", "", punctuation))


df['length'] = list(map(lambda x: len(str(x).split()), df['review']))
df = df.drop_duplicates(subset=["condition" ,"review", "rating"]).reset_index(drop=True)
df.isnull().any()
drugC = df.drugName.value_counts()
drugC = drugC[drugC>=5]
df = df.loc[df['drugName'].isin(drugC.index),]
condC = df.condition.value_counts()
condC = condC[condC>=5]
df = df.loc[df['condition'].isin(condC.index),]
df = df.loc[df['usefulCount']>6,]
df.loc[:,'condition'] = df['condition'].apply(lambda x: 'unknown' if re.search("users found",x) else str(x).lower())
df.loc[:,'drugName'] = df['drugName'].apply(lambda x: str(x).lower())
df.loc[:,'review'] = df['review'].apply(lambda x: str(x).lower())
df.review = df.review.str.lower()
df["condition"].fillna("unknown", axis=0, inplace=True)

df["condition"].nunique()
df["review"] = df.review.str.replace('"', "")
df["review"] = df.review.str.replace('&#039;', "")
df.review = df.review.str.replace(r'[^\x00-\x7F]+',' ')
#df.review = df.review.str.replace(r'^\s+|\s+?$','')
df.review = df.review.str.replace(r'\s+',' ')
df.review = df.review.str.replace(r'\.{2,}', '')
df.review = df.review.str.replace(r'\d+', ' ')
df.review = df.review.str.replace(r"\s*'\s*\w*", ' ')
df.review = df.review.str.replace(r'\W+', ' ')
df.review = df.review.str.replace(r'\s+', ' ')
df.review = df.review.str.replace(r'^\s+|\s+?$', '')
df
stop_spec = ['taking','pain','effects','first','started','like','months','get','days','time','would','one','weeks','took','week','also','got','month']
stop_spec.extend(['day','years','life','went','year','hours','going','used','lbs','getting','try','use','make','say'])
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
from sklearn.feature_extraction import text
stop = text.ENGLISH_STOP_WORDS
stop_words.extend(['im', 'ive', 'it', 'mg', 'quot'])
stop_words.extend(stop)
stop_words.extend(stop_spec)
stop_words = list(set(stop_words))
for i in range(len(stop_words)):
    stop_words[i] = re.sub("'","",stop_words[i])
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
pat

df['review'] = df['review'].str.replace(pat, '')
df.review = df.review.str.replace(r'\W+', ' ')
reviews = []
corpus=[]
for review in df['review']:
    reviews.append(review)
    corpus.append(nltk.sent_tokenize(review))
corpus=[sent for sublist in corpus for sent in sublist]
wordfreq = {}
for sentence in corpus:
    words = sentence.split()
    #tokens = nltk.word_tokenize(sentence) # To get the words, it can be also done with sentence.split()
    for word in words:
        if ( word not in wordfreq.keys() ): ## first time appearnce in the sentence
            wordfreq[word] = 1 # We initialize the corresponding counter
        else: ## if the world is already existed in the dictionalry 
            wordfreq[word] += 1 # We increase the corresponding counter
wordfreq = dict(sorted(wordfreq.items(),key= lambda x:x[1],reverse=True))
print(wordfreq)

len(list(wordfreq.keys()))
# Keeping 30 most preq words
corpus_freq = [(wordfreq[key],key) for key in list(wordfreq.keys())]
corpus_freq = [(word[1],word[0]) for word in corpus_freq[:60]] 

from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
lem = WordNetLemmatizer()
corpus_freq = [(lem.lemmatize(word[0]),word[1]) for word in corpus_freq]
np.array(list(cols.keys()))

cols = {word[0]: [] for word in corpus_freq}
reviews = pd.DataFrame(cols)
reviews.columns

def review_inpector(sentence, words):
    # Initializing an empty dictionary of word frequencies for the corresponding review
    tokens = nltk.word_tokenize(sentence)
    col_freq = {col:0 for col in words}
    # Filling the dictionary with word frequencies in the review
    for token in tokens:
        if token in words:
            col_freq[token] += 1

    return col_freq
my_list = list(map(review_inpector, df['review'],[list(cols.keys())]*df.shape[0] ) )
my_list[:2]
reviews = pd.DataFrame(my_list)
reviews['rating'] = df['rating'].reset_index(drop=True)
reviews
