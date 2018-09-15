# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 17:06:43 2018

@author: admin
"""


import jieba
from wordcloud import WordCloud
import pandas as pd
import re
#df = pd.read_csv("datascience.csv", encoding='gb18030')
df = pd.read_csv("xiaoyan.csv", encoding='gb18030')
df.head()
def process(mytext):
    return re.sub(u'[^0-9a-zA-Z\u4e00-\u9fa5]+', '', mytext)
df['content1'] = df.content.apply(process)

a = ''.join(df['content1'])
#with codecs.open('pjl_comment.txt',encoding='utf-8') as f:
 #   comment_text = f.read()
#cut_text = " ".join(jieba.cut(comment_text)) # 将jieba分词得到的关键词用空格连接成为字符串

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
df["content_cutted"] = df.content1.apply(chinese_word_cut)



mytext = df.content_cutted[2]

mytext1 = ' '.join(df.content_cutted) 
#mytext1 = re.sub(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”'']+', '', mytext)
wordcloud = WordCloud(font_path="simsun.ttf").generate(mytext1)

wordcloud.to_file("pjl_cloud.tiff")

import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
