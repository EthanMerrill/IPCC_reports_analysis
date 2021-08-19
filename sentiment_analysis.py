#%%
import re
import os
import datetime as dt
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tika import parser
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

nltk.download(['stopwords','punkt', 'vader_lexicon'])
#%%
raw = parser.from_file('IPCC_reports/IPCC_1992.pdf')
# print(raw['content'])

#%%
# Tokenize the words
# print(nltk.word_tokenize(raw['content'])[:525])
# Remove the punctuation 
words = [w for w in nltk.word_tokenize(raw['content']) if w.isalpha()]
#%%
# Remove stopwords from the text:
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words if w.lower() not in stopwords and len(w)>2]

#%%
# create frequency distribution of these words
fd = nltk.FreqDist(words)
fd.most_common(50)
fd.tabulate(5)

# %%
# run frequencies on bigrams, trigrams
trigrams = nltk.collocations.TrigramCollocationFinder.from_words(words)
trigrams.ngram_fd.most_common(20)
#%%

quadgrams = nltk.collocations.QuadgramCollocationFinder.from_words(words)
quadgrams.ngram_fd.most_common(20)
# %%
# Sentiment analysis

sentences = nltk.sent_tokenize(raw['content'])

# filter out just period sentences:
sentences = list(filter(lambda i:i!='.', sentences))
# use regular expressions to seek and destroy new line characters
i = 0
for s in list(sentences):
    sentences[i] = re.sub('\\n',"", s)
    i=i+1

# %%
sia = SentimentIntensityAnalyzer()

# %%
sentenceSentiment_df = pd.DataFrame()

for i in sentences:
    sentenceSentiment_df= sentenceSentiment_df.append(sia.polarity_scores(i),ignore_index=True)

sentenceSentiment_df = sentenceSentiment_df.mean().to_frame().transpose()
sentenceSentiment_df['year'] = pd.to_datetime('2014')
print(sentenceSentiment_df.median().to_frame().transpose())

# %%
## Functionalize the sentiment analyizer so it takes the pdf and year as an argument and returns a df with the mean sentiment
def analyze_pdf_sentiment(yearString, pdf):
    # tokenize the sentences
    sentences = nltk.sent_tokenize(pdf['content'])
    # filter out just period sentences:
    sentences = list(filter(lambda i:i!='.', sentences))
    # use regular expressions to seek and destroy new line characters
    i = 0
    for s in list(sentences):
        sentences[i] = re.sub('\\n',"", s)
        i=i+1

    sia = SentimentIntensityAnalyzer()
    sentenceSentiment_df = pd.DataFrame()
    for i in sentences:
        sentenceSentiment_df= sentenceSentiment_df.append(sia.polarity_scores(i),ignore_index=True)
        # print(i,sentenceSentiment_df)
    sentenceSentiment_df = sentenceSentiment_df.mean().to_frame().transpose()
    sentenceSentiment_df['year'] = pd.to_datetime(yearString)
    # print(f'year complete: {sentenceSentiment_df['year']} ')
    return (sentenceSentiment_df)

# %% 
# test the analyize pdf sentiment function
analyze_pdf_sentiment('2029', parser.from_file('IPCC_reports/IPCC_1990.pdf'))

# %%

# %%
#function takes a list of filenames, then makes them into pdfs and extracts the years
def pdfER(report_dir):
    file_list = os.listdir(report_dir)
    print(file_list)
    pdf_objects = {}
    for file in file_list:
        year = re.search('\d\d\d\d', file).group(0)
        pdf_objects[year] = parser.from_file('IPCC_reports/'+file)

    return pdf_objects


def justWords(pdf ):
    # Remove the punctuation 
    words = [w for w in nltk.word_tokenize(pdf['content']) if w.isalpha()]
    #%%
    # Remove stopwords from the text:
    stopwords = nltk.corpus.stopwords.words('english')
    words = [w for w in words if w.lower() not in stopwords and len(w)>2]
    words =list(map(lambda x: x.lower(), words))
    return words

# %%

pdf_objects = pdfER('IPCC_reports')
historical_sentiment_df = pd.DataFrame()
for key, value in pdf_objects.items():
    historical_sentiment_df = historical_sentiment_df.append(analyze_pdf_sentiment(key, value),ignore_index=True)
    print(historical_sentiment_df)
    # print(key, type(value['content']))
historical_sentiment_df.to_csv('sentiment.csv')

# %%
analyze_pdf_sentiment('2021', parser.from_file('IPCC_reports/IPCC_1990.pdf'))

# %%
likelyhood_scale = ['virtually certain', 'very likely', 'likely', 'about as likely as not', 'unlikely', 'very unlikely', 'exceptionally unlikely']

words2021full = justWords(parser.from_file('IPCC_full_reports/IPCC_2021_full.pdf'))
# %%
counts = nltk.FreqDist(words2021full)

selected_words_freq = [counts[x] or 0 for x in likelyhood_scale]
# %%
# sentences = nltk.sent_tokenize(parser.from_file('IPCC_reports/IPCC_1990.pdf')['content'])
bigrams = nltk.collocations.BigramCollocationFinder.from_words(words2021full)
# bigramfreq1990 = nltk.FreqDist(bigrams)
# %%


sentences = nltk.sent_tokenize(parser.from_file('IPCC_reports/IPCC_1990.pdf')['content'])
# filter out just period sentences:
sentences = list(filter(lambda i:i!='.', sentences))
# use regular expressions to seek and destroy new line characters
i = 0
for s in list(sentences):
    sentences[i] = re.sub('\\n',"", s)
    i=i+1


# %%
fullText = justWords(parser.from_file('IPCC_reports/IPCC_2021_Full.pdf'))
#%%
bigrams = nltk.collocations.BigramCollocationFinder.from_words(words2021full)
trigrams = nltk.collocations.TrigramCollocationFinder.from_words(words2021full)
#%%
# likely filter
likely_filter = lambda *w: 'certain' not in w
trigrams.apply_ngram_filter(likely_filter)
bigrams.apply_ngram_filter(likely_filter)
bigrams.ngram_fd.most_common(50)
trigrams.ngram_fd.most_common(50)
# %%
