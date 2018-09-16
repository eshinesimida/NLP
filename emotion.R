library(dplyr)
library(tidytext)
library(tidyr)
library(ggplot2)

script <- read.csv("data.csv", stringsAsFactors=FALSE)
names(script) <- c('line', 'text')

tidy_script <- script %>%  unnest_tokens(word, text)
head(tidy_script)

tidy_script %>%  inner_join(get_sentiments("nrc")) %>%  arrange(line) %>%  head(10)

tidy_script %>%  inner_join(get_sentiments("nrc")) %>%  count(line, sentiment) %>%  arrange(line) %>%  head(10)
tidy_script %>%  inner_join(get_sentiments("nrc")) %>%  
  count(line, sentiment) %>%  mutate(index = line %/% 5) %>%  arrange(index) %>%  head(10)

t1 <- tidy_script %>%  inner_join(get_sentiments("nrc"))
t2 <- t1 %>% count(line,sentiment)
t2 %>% mutate(index = as.numeric(line) %/% 5) %>%  arrange(index) %>%  head(10)

tidy_script %>%  inner_join(get_sentiments("nrc")) %>%  count(line, sentiment) %>%  mutate(index = as.numeric(line) %/% 5) %>%  ggplot(aes(x=index, y=n, color=sentiment)) %>%  + geom_col()## Joining, by = "word"


tidy_script %>%  inner_join(get_sentiments("nrc")) %>%  count(line, sentiment) %>%  mutate(index = as.numeric(line) %/% 5) %>%  ggplot(aes(x=index, y=n, color=sentiment)) %>%  + geom_col() %>%  + facet_wrap(~sentiment, ncol=3)## Joining, by = "word"

tidy_script %>%  inner_join(get_sentiments("nrc")) %>%  filter(sentiment == "positive") %>%  count(word) %>%  arrange(desc(n)) %>%  head(10)

tidy_script %>%  inner_join(get_sentiments("nrc")) %>%  filter(sentiment == "negative") %>%  count(word) %>%  arrange(desc(n)) %>%  head(10)

tidy_script %>%  anti_join(stop_words) %>%  inner_join(get_sentiments("nrc")) %>%  filter(sentiment == "positive") %>%  count(word) %>%  arrange(desc(n)) %>%  head(10)

custom_stop_words <- bind_rows(stop_words,
                               data_frame(word = c("stark", "mother", "father", "daughter", "brother", "rock", "ground", "lord", "guard", "shoulder", "king", "main", "grace", "gate", "horse", "eagle", "servent"),lexicon = c("custom")))

tidy_script %>%  anti_join(custom_stop_words) %>%  inner_join(get_sentiments("nrc")) %>%  filter(sentiment == "positive") %>%  count(word) %>%  arrange(desc(n)) %>%  head(10)

tidy_script %>%  anti_join(custom_stop_words) %>%  inner_join(get_sentiments("nrc")) %>%  filter(sentiment == "negative") %>%  count(word) %>%  arrange(desc(n)) %>%  head(10)

tidy_script %>%  anti_join(custom_stop_words) %>%  inner_join(get_sentiments("nrc")) %>%  filter(sentiment != "negative" & sentiment != "positive") %>%  count(line, sentiment) %>%  mutate(index = as.numeric(line) %/% 5) %>%  ggplot(aes(x=index, y=n, color=sentiment)) %>%  + geom_col() %>%  + facet_wrap(~sentiment, ncol=3)
