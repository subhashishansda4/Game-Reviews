# LIBRARIES
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import os

import re
import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')

# https://pypi.org/project/syllables/
import syllables

from nltk.stem import WordNetLemmatizer,PorterStemmer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

from transformers import pipeline
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------------------

# RAW DATAFRAME
df = pd.read_csv('output/raw_df.csv', encoding='latin-1')
sample_size = 2000
raw_df = df.sample(sample_size, random_state=1)

# ------------------------------------------------------------------------

# OVERVIEW
print('INFO:')
print(raw_df.info())
print('')
print('')

# missing values
print('MISSING VALUES:')
print('null values')
print(raw_df.isnull().sum())
print('')
# duplicate values
print('duplicate values')
print(raw_df.duplicated().sum())
print('')

raw_df.dropna(axis=0, inplace=True)

# ------------------------------------------------------------------------

# INITIAL EDA
# number of positives
sns.countplot(data=raw_df, x='POSITIVE')
plt.savefig('graphs/positive.jpg')

# number of positives per game
plt.figure(figsize = (8, 5), facecolor = None)
plt.tight_layout(pad=0)
sns.countplot(data=raw_df, x='GAME', hue='POSITIVE', palette='magma')
plt.savefig('graphs/positive_per_game.jpg')

# scatterplot for votes and score as per positives
sns.scatterplot(data=raw_df, x='VOTES', y='SCORE', hue='POSITIVE', palette='Blues')
plt.savefig('graphs/votes_&_score.jpg')

# countplots for days, months and years
sns.countplot(data=raw_df, x='YEAR', palette='PuOr')
plt.savefig('graphs/year.jpg')
sns.countplot(data=raw_df, x='MONTH', palette='PuOr')
plt.savefig('graphs/month.jpg')
sns.countplot(data=raw_df, x='DAY', palette='PuOr')
plt.savefig('graphs/day.jpg')

# ------------------------------------------------------------------------

# STOP WORDS
stop_type = ['Auditor', 'Currencies', 'DatesandNumbers', 'Generic', 'GenericLong', 'Geographic', 'Names']
stop_words_list = []
stop_file = 'data/StopWords/'

for file in tqdm(os.listdir(stop_file)):
    path = os.path.join(stop_file, file)
    
    if path == os.path.join(stop_file, 'StopWords_GenericLong.txt'):
        with open(path, 'r') as f:
            stop_words_list.append(f.read().split())
    else:
        with open(path, 'r') as f:
            words = f.read().split()
            upper_words = [word for word in words if word.isupper()]
            upper_words = [words.lower() for words in upper_words]
            stop_words_list.append(upper_words)

for i in range(len(stop_type)):
    stop_words_list[i] = ' '.join(stop_words_list[i])
stop_words_list = ' '.join(stop_words_list)

# 1 SW-LIST
stops1 = stop_words_list.split()
# 2 SW-LIST
stops2 = list(stopwords.words('english'))
# 3 SW-LIST
stops3 = list(STOPWORDS)
# main stopwords
STOPS = set(stops1 + stops2 + stops3)

# -----------------------------------------------------------------------------

# POSITIVE AND NEGATIVE
negative_words = []
positive_words = []
value_file = 'data/MasterDictionary/'

for file in tqdm(os.listdir(value_file)):
    path = os.path.join(value_file, file)
    if file == 'negative-words.txt':
        with open(path, 'r') as f:
            negative_words.append(f.read())       
    else:
        with open(path, 'r') as f:
            positive_words.append(f.read())

negative_words = ' '.join(negative_words)
negative_words = negative_words.split()
negative_words = [words.lower() for words in negative_words]

positive_words = ' '.join(positive_words)
positive_words = positive_words.split()
positive_words = [words.lower() for words in positive_words]

# -----------------------------------------------------------------------------

# removing stopwords
def stop_words(s):
    words = nltk.word_tokenize(s)
    clean = [word for word in words if word not in STOPS]
    
    s = ' '.join(clean)
    return s

# number to words
def number_to_words(num):
    to_19 = 'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()
    
    def words(n):
        if num == 0:
            return 'zero'
        if n < 20:
            return to_19[n-1:n][0]
        if n < 100:
            return tens[n//10-2] + ('' if n%10==0 else ' ' + words(n%10))
        if n < 1000:
            return to_19[n//100-1] + ' hundred' + ('' if n%100==0 else ' and ' + words(n%100))
        for p, w in enumerate(('thousand', 'million', 'billion'), 1):
            if n < 1000**(p+1):
                return words(n//1000**p) + ' ' + w + ('' if n%1000**p==0 else ' ' + words(n%1000**p))
    
    return words(num)

def replace_numbers_with_words(text):
    for word in re.findall(r'\b\d+\b', text):
        num = int(word)
        new_word = number_to_words(num)
        try:
            text = text.replace(word, new_word)
        except:
            ''
    return text

# -----------------------------------------------------------------------------

# PROCESS
def process(s):
    s = s.lower()
    
    # replace certain special characters with their string esuivalents
    s = s.replace('%', ' percent ')
    s = s.replace('$', ' dollar ')
    s = s.replace('₹', ' rupee ')
    s = s.replace('€', ' euro ')
    s = s.replace('@', ' at ')
    
    # remove remaining special characters
    special = re.compile(r'[^A-Za-z0-9]+')
    s = re.sub(special, ' ', s).strip()
    
    # replace connected words with full words
    s = s.replace("'ve", " have")
    s = s.replace("'t", " not")
    s = s.replace("'re", " are")
    s = s.replace("'ll", " will")
    s = s.replace("'m", " am")
    s = s.replace("'s", " ")
        
    # remove punctuations
    pattern = re.compile('[^\w\s]')
    s = re.sub(pattern, ' ', s).strip()
    
    # replacing numbers with string esuivalents
    s = replace_numbers_with_words(s)
    
    s = ' '.join(s.split())
    
    return s


stems = []
lemms = []

# CLEAN
def clean(s):
    s = process(s)
    # remove stopwords
    s = stop_words(s)
    
    # stemming and lemmatization
    stem_words = [stemmer.stem(w) for w in s]
    lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
    
    stems.append(''.join(stem_words))
    lemms.append(''.join(lemma_words))
    
    s = ' '.join(s.split())
    return s
    


process_df = pd.DataFrame(raw_df, columns=['DATE', 'DAY', 'MONTH', 'YEAR', 'TIME', 'GAME', 'RAW'])
process_df['PROCESS'] = process_df['RAW'].apply(process)
process_df['CLEAN'] = process_df['RAW'].apply(clean)
process_df['LEMMS'] = lemms
process_df['POSITIVE'] = raw_df['POSITIVE']
process_df['VOTES'] = raw_df['VOTES']
process_df['SCORE'] = raw_df['SCORE']
process_df.to_csv('output/process_df.csv')

# -----------------------------------------------------------------------------

# FEATURE ENGINEERING
def positive_negative(words, negative_score, positive_score):
    for word in words:
        if word in positive_words:
            positive_score += 1
        elif word in negative_words:
            negative_score += 1
    return [negative_score, positive_score]

def polarity(words, polarity_score, negative_score, positive_score):    
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    polarity_score = max(-1, min(polarity_score, 1))
    return round(polarity_score, 2)

def subjectivity(words, subjectivity_score, negative_score, positive_score):
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    subjectivity_score = max(0, min(subjectivity_score, 1))
    return round(subjectivity_score, 2)

def average_word_length(characters_count, words_count, word_len):
    try:
        word_len = (characters_count / words_count)
    except:
        word_len = 0
    return round(word_len, 2)

def personal_pronouns(text, pronouns, pronoun_len):
    pronoun = pronouns.findall(text)
    pronoun_len = len(pronoun)
    return pronoun_len

def syllable_count(words, sentences_count, words_count, syllable_len):
    for word in words:
        syllable_len += syllables.estimate(word)
        
    syllable_len -= sentences_count
    try:
        syllable_len = (syllable_len / words_count)
    except:
        syllable_len = 0
    return round(syllable_len, 2)

def complex_word_count(words, complex_word_len):
    for word in words:
        if syllables.estimate(word) > 2:
            complex_word_len += 1
    return complex_word_len

def average_word_per_sentence(words_count, sentences_count, word_per_sentence_len):
    try:
        word_per_sentence_len = (words_count / sentences_count)
    except:
        word_per_sentence_len = 0
    return round(word_per_sentence_len, 2)

def percent_complex_count(words, words_count, percent_complex_len):
    try:
        percent_complex_len = (complex_word_count(words, 0) / words_count)
    except:
        percent_complex_len = 0
    return round(percent_complex_len, 2)

def fog_index(words, words_count, sentences_count, fog):
    fog = 0.4 * (average_word_per_sentence(words_count, sentences_count, 0) + percent_complex_count(words, words_count, 0))
    return round(fog, 2)


p_scores = []
n_scores = []
pol_scores = []
sub_scores = []
word_scores = []

syllable_scores = []
word_len_scores = []
complex_word_scores = []
word_per_sen_scores = []
percent_complex_scores = []
fog_scores = []

pnoun_scores = []

lemms_scores = []


texts = process_df['CLEAN']
for text in texts:
    words = nltk.word_tokenize(text)
    
    word_count = len(words)
    
    n_scores.append(positive_negative(words, 0, 0)[0])
    p_scores.append(positive_negative(words, 0, 0)[1])
    word_scores.append(word_count)
    
for i in range(0, 1991):
    pol_scores.append(polarity(words, 0, n_scores[i], p_scores[i]))
    sub_scores.append(subjectivity(words, 0, n_scores[i], p_scores[i]))

    
texts_ = process_df['PROCESS']
for text in texts_:
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    
    characters_count = len(text) - len(sentences)
    sentences_count = len(sentences)
    words_count = len(words) - len(sentences)
    
    word_len_scores.append(average_word_length(characters_count, words_count, 0))
    syllable_scores.append(syllable_count(words, sentences_count, words_count, 0))
    complex_word_scores.append(complex_word_count(words, 0))
    word_per_sen_scores.append(average_word_per_sentence(words_count, sentences_count, 0))
    percent_complex_scores.append(percent_complex_count(words, words_count, 0))
    fog_scores.append(fog_index(words, words_count, sentences_count, 0))

texts__ = process_df['LEMMS']
for text in texts__:
    words__ = nltk.word_tokenize(text)
    
    lemms_count = len(words__)
    lemms_scores.append(lemms_count)

# personal pronouns
pronouns = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
texts__ = process_df['RAW']
for text in texts__:
    pnoun_scores.append(personal_pronouns(text, pronouns, 0))


# sentiment analysis
sentiments = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

senm_scores = []
texts = process_df['PROCESS']
for text in texts:
    words = nltk.word_tokenize(text)
    a = sentiments(words)
    try:
        a = sentiments(words)
        
        if a[0]['label'] == 'POSITIVE':
            senm = a[0]['score']
        elif a[0]['label'] == 'NEGATIVE':
            senm = -a[0]['score']
        
        senm_scores.append(round(senm, 4))
    except:
        senm_scores.append(0)


# hashtag generation
OPTIONS = ['gameplay', 'controls', 'sound', 'graphics']
hashtags = []
tags = []

sens = list(process_df['CLEAN'])
wrds = [s.split() for s in sens]

w2v = Word2Vec(wrds, vector_size=10, window=5, min_count=0, workers=4)
w2v.build_vocab(wrds, update=True)
w2v.train(wrds, total_examples=len(sens), epochs=10)
w2v.save('word2vec.model')

texts = process_df['CLEAN']
for text in texts:
    words = nltk.word_tokenize(text)
    model = Word2Vec.load('word2vec.model')
    
    n = 0
    for word in words:
        for option in OPTIONS:
            similarity = model.wv.similarity(word, option)
            if similarity > n:
                n = similarity
                hashtags.append(OPTIONS.index(option))
    tags.append(hashtags[len(hashtags)-1])


feat_df = pd.DataFrame(process_df)
feat_df['POSITIVE SCORE'] = p_scores
feat_df['NEGATIVE SCORE'] = n_scores
feat_df['POLARITY SCORE'] = pol_scores
feat_df['SUBJECTIVITY SCORE'] = sub_scores
feat_df['LENGTH'] = process_df['PROCESS'].str.len()
feat_df['AVG SENTENCE LENGTH'] = word_per_sen_scores
feat_df['PERCENTAGE OF COMPLEX WORDS'] = percent_complex_scores
feat_df['FOG INDEX'] = fog_scores
feat_df['AVG NUMBER OF WORDS PER SENTENCE'] = word_per_sen_scores
feat_df['COMPLEX WORD COUNT'] = complex_word_scores
feat_df['WORD COUNT'] = word_scores
feat_df['SYLLABLE PER WORD'] = syllable_scores
feat_df['PERSONAL PRONOUNS'] = pnoun_scores
feat_df['AVG WORD LENGTH'] = word_len_scores
feat_df['LEMMS SCORE'] = lemms_scores
feat_df['SENTIMENT'] = senm_scores
feat_df['TAGS'] = tags

feat_df.to_csv('output/features.csv')

# -----------------------------------------------------------------------------

# PROCESSED EDA
# word-cloud
wordcloud = WordCloud(width=1080, height=1920, background_color='white', stopwords=STOPS, min_font_size=10).generate(' '.join(process_df['PROCESS']))
plt.figure(figsize = (15, 8), facecolor = None)
plt.imshow(wordcloud)
plt.tight_layout(pad=0)
plt.axis('off')
plt.savefig('graphs/word_cloud_3.jpg')
plt.show()

# polarity score per game
sns.kdeplot(x=feat_df['POLARITY SCORE'], hue=feat_df['GAME'])
plt.savefig('graphs/polarity_score.jpg')

# number of positives per game
sns.kdeplot(data=feat_df, x='SUBJECTIVITY SCORE', hue='POSITIVE', fill=True)
plt.savefig('graphs/subjectivity_score.jpg')

# ---------------------------------------------------------------------

# MACHINE LEARNING
# vectorization
tf_idf = TfidfVectorizer(max_features=100)

arr = []
for s in sens:
    ar = []
    for w in nltk.word_tokenize(s):
        try:
            ar.append(w2v.wv[w])
        except:
            pass
    arr.append(ar)
    
sample = []
for arr in arr:
    s = ', '.join(str(x) for x in arr)
    s = s.replace('[', '').replace(']', '')
    sample.append(s)
    
sens_arr = tf_idf.fit_transform(sample).toarray()

vect_df = pd.DataFrame(sens_arr, index=feat_df.index)
vect_df = pd.concat([feat_df, vect_df], axis=1)
vect_df.columns = vect_df.columns.astype(str)
vect_df.to_csv('output/vect_df.csv')



# training and testing datasets
train_df = vect_df[vect_df['GAME'] != 'Destiny 2']
train_df = train_df.drop(columns=['DATE', 'DAY', 'MONTH', 'YEAR', 'TIME', 'GAME', 'RAW', 'PROCESS', 'CLEAN', 'LEMMS'])
x_train = train_df.drop(columns=['TAGS', 'POSITIVE'])

y_train_tags = pd.DataFrame(train_df, columns=['TAGS'])
y_train_tags = y_train_tags.values.ravel()
y_train_sent = pd.DataFrame(train_df, columns=['POSITIVE'])
y_train_sent = y_train_sent.values.ravel()

test_df = vect_df[vect_df['GAME'] == 'Destiny 2']
test_df = test_df.drop(columns=['DATE', 'DAY', 'MONTH', 'YEAR', 'TIME', 'GAME', 'RAW', 'PROCESS', 'CLEAN', 'LEMMS'])
x_test = test_df.drop(columns=['TAGS', 'POSITIVE'])

y_test_tags = pd.DataFrame(test_df, columns=['TAGS'])
y_test_tags = y_test_tags.values.ravel()
y_test_sent = pd.DataFrame(test_df, columns=['POSITIVE'])
y_test_sent = y_test_sent.values.ravel()

# model selection
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


lr = LogisticRegression()  
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lda = LinearDiscriminantAnalysis()
svc = SVC(probability=True)
nb = GaussianNB()

models = [dt, rf, lda, svc, nb]
param_grid = {}


scrs = []
for model in models:
    grid = GridSearchCV(
        model,
        param_grid,
        scoring='neg_log_loss', refit='neg_log_loss'
    )
    
    print(model)
    
    # tags
    grid.fit(x_train, y_train_tags)
    scr = grid.score(x_test, y_test_tags)
    print("score : {:.2f}".format(np.mean(scr)))
    scrs.append(scr)
    
    # sentiment
    grid.fit(x_train, y_train_sent)
    scr = grid.score(x_test, y_test_sent)
    print("score : {:.2f}".format(np.mean(scr)))
    scrs.append(scr)
    
    grid = GridSearchCV(
        model,
        param_grid,
        scoring='roc_auc', refit='roc_auc'
    )
    

    
    # tags
    grid.fit(x_train, y_train_tags)
    y_pred = grid.predict(x_test)
    report = classification_report(y_test_tags, y_pred, labels=range(len(OPTIONS)), target_names=OPTIONS)
    print(report)
    
    mtrx = metrics.confusion_matrix(y_test_tags, y_pred)
    print("confusion matrix")
    print(mtrx)

    # sentiment
    print('')
    grid.fit(x_train, y_train_sent)
    y_pred = grid.predict(x_test)
    report = classification_report(y_test_sent, y_pred, labels=range(len(OPTIONS)), target_names=OPTIONS)
    print(report)
    
    mtrx = metrics.confusion_matrix(y_test_sent, y_pred)
    print("confusion matrix")
    print(mtrx)
    print("")
    print("")
    print("")
    print("")


# Random Forest
# tags
rf_cls = RandomForestClassifier(n_estimators=100, random_state=0)
rf_cls.fit(x_train, y_train_tags)
y_pred = rf_cls.predict(x_test)

out_tags = pd.DataFrame({'TAGS':y_test_tags, 'pred_tags':y_pred})
print(out_tags)
print(out_tags.describe())

# sent
rf_cls = RandomForestClassifier(n_estimators=100, random_state=0)
rf_cls.fit(x_train, y_train_sent)
y_pred = rf_cls.predict(x_test)

out_sent = pd.DataFrame({'SENT':y_test_sent, 'pred_sent':y_pred})
print(out_sent)
print(out_sent.describe())


# FINAL OUTPUT
label_mapping = {0:'Gameplay', 1:'Controls', 2:'Sound', 3:'Graphics'}
out_tags['TAGS'] = out_tags['TAGS'].map(label_mapping)
out_tags['pred_tags'] = out_tags['pred_tags'].map(label_mapping)

OUTPUT = pd.concat([out_tags, out_sent], axis=1)
OUTPUT.to_csv('output/final_df.csv', index=False)

# FINAL EDA
sns.countplot(data=OUTPUT, x='TAGS')
plt.savefig('graphs/output_tags.jpg')
sns.countplot(data=OUTPUT, x='pred_tags')
plt.savefig('graphs/output_pred_tags.jpg')

sns.countplot(data=OUTPUT, x='SENT')
plt.savefig('graphs/output_sent.jpg')
sns.countplot(data=OUTPUT, x='pred_sent')
plt.savefig('graphs/output_pred_sent.jpg')