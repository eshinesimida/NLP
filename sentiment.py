# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:28:44 2018

@author: admin
"""

#sentiment-analysis

text = 'I am happy today. I feel sad today.'
from textblob import TextBlob
blob = TextBlob(text)
blob
blob.sentences
blob.sentences[0].sentiment
blob.sentences[1].sentiment
blob.sentiment

text = u'我今天很快乐。我今天很愤怒。'

from snownlp import SnowNLP
s = SnowNLP(text)
for sentence in s.sentences:
    print(sentence)
    
s1 = SnowNLP(s.sentences[0])
s1.sentiments

s2 = SnowNLP(s.sentences[1])
s2.sentiments