# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:58:44 2018

@author: admin
"""
import jieba
import pandas as pd
df = pd.read_csv("datascience.csv", encoding='gb18030')
df = pd.read_csv("xiaoyan.csv", encoding='gb18030')
df.head()

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
df["content_cutted"] = df.content.apply(chinese_word_cut)

df.content_cutted.head()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_features = 1000

tf_vectorizer = CountVectorizer(strip_accents = 'unicode', max_features=n_features, stop_words='english', max_df = 0.5, min_df = 10) 
tf = tf_vectorizer.fit_transform(df.content_cutted)

from sklearn.decomposition import LatentDirichletAllocation

n_topics = 5
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)

LatentDirichletAllocation(batch_size=128, doc_topic_prior=None, 
                          evaluate_every=-1, learning_decay=0.7, 
                          learning_method='online', 
                          learning_offset=50.0,
                          max_doc_update_iter=100, 
                          max_iter=50, mean_change_tol=0.001, 
                          n_jobs=1, n_topics=5, perp_tol=0.1, 
                          random_state=0, topic_word_prior=None, 
                          total_samples=1000000.0, verbose=0)

def print_top_words(model, feature_names, n_top_words): 
    for topic_idx, topic in enumerate(model.components_): 
        print("Topic #%d:" % topic_idx) 
        print(" ".join([feature_names[i] 
              for i in topic.argsort()[:-n_top_words - 1:-1]])) 
    print()

n_top_words = 20

tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.show(data)

##################
n_topics = 10 
lda = LatentDirichletAllocation(n_topics=n_topics, 
                                max_iter=50, learning_method='online', 
                                learning_offset=50., 
                                random_state=0) 
lda.fit(tf) 
print_top_words(lda, tf_feature_names, n_top_words) 
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.show(data)

##############################
n_topics = 18 
lda = LatentDirichletAllocation(n_topics=n_topics, 
                                max_iter=50, learning_method='online', 
                                learning_offset=50., 
                                random_state=0) 
lda.fit(tf) 
print_top_words(lda, tf_feature_names, n_top_words) 
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.show(data)


