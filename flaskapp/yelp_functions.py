from flask import Flask, render_template, request
import requests
import pandas as pd
import re
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import nltk
from nltk import pos_tag
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import string
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

def parse_for_word(row, keyword):
    has_keyword = 0
    text = row["Review"]
    #print (text + "\n")
    average_sentiment = 0
    for word in text.split(" "):
        if keyword in word:
            has_keyword = 1
    if (has_keyword > 0):
        sentence_list, average_sentiment = get_sentences_with_word(text, keyword)
    else:
        sentence_list = []
        combined_score = 0
    return has_keyword, sentence_list, average_sentiment

def get_sentences_with_word(text, keyword):
    average_sentiment = 0
    number_of_sentences = 0
    list_of_sentences = []
    #print(text)
    sentences = tokenize.sent_tokenize(text)
    for sentence in sentences:
        #print (sentence)
        if keyword in sentence:
            list_of_sentences.append(sentence)
            number_of_sentences = number_of_sentences + 1
            sentiment = get_sentiment(sentence)
            #print(sentence, sentiment)
    average_sentiment = float(sentiment / number_of_sentences)
    return list_of_sentences, average_sentiment

def get_sentiment(text):
    sentiment = SentimentIntensityAnalyzer() #### calling Intensity Analyzer
    compound = sentiment.polarity_scores(text)['compound']  ### calling the 'compound' score for the "text" entered
    #if compound > 0:
    #    return 1  ## positive
    #else:
    #    return 0 ## negative
    #else:
        #return "Neutral"
    #print(compound)
    return compound

def scaled_combined_score(row):
    # get the values
    sentiment_score = float(row["average_sentiment_sentence_average"])
    has_keyword_sum = int(row["has_keyword_sum"])
    sentiment_vader = float(row["sentiment_vader_average"])
    # update the sentiment value
    return (combined_function(has_keyword_sum, sentiment_score, sentiment_vader))

def combined_function(has_keyword_sum, sentiment_score, sentiment_vader):
    weight = 0
    if (has_keyword_sum >= 3):
        weight = 0.7
    else:
        alpha = -0.07777
        beta = 0.46666
        weight = alpha * has_keyword_sum * has_keyword_sum + beta * has_keyword_sum
        #print(weight)
    #print(weight, sentiment_score, sentiment_vader)
    return weight * sentiment_score + 0.3 * sentiment_vader
