---
date: "2018-09-09T00:00:00Z"
draft: false
lastmod: "2018-09-09T00:00:00Z"
linktitle: Unsupervised Learning
menu:
  mad:
    name: Unsupervised Learning
    weight: 1
summary: Unsupervised Learning for M2 Data Science Evry
title: Unsupervised Learning 
toc: true
type: docs
weight: 1
---


## Outline

1. Discrete latent factor 
  - Hidden markov models (example in genetics)
  - Variational EM (example of Stochastic block model)
2.  Continuous Latent Variable 
  - Independant Component Analysis


## Reference document
The lecture closely follows and largely borrows material from 
"Machine Learning: A Probabilistic Perspective" (MLAPP) from
Kevin P. Murphy, chapters:

  - Chapter 17: Markov and hidden Markov models
  - Chapter 21: Variational inference
  - Chapter 12:  Latent Linear Models
  - Chapter 13: Sparse Linear Models


## Lectures Notes

  - [Lecture 1 (Hidden Markov Models)](media/unsupervised-M2-data-science.pdf)  


## Exercices

### Unigrams and bigrams

Using the song of Leonard Cohen ['Suzanne'](media/suzanne-cohen-eng.txt) compute the unigrams  and bigrams considering the letters of the alphabet as the states of the chain. 

```{r}
suzanne_eng <- readLines("suzanne-cohen-eng.txt")
unigrams <- suzanne_eng %>% paste(collapse=" ") %>% 
  # tokenize by character (strsplit returns a list, so unlist it)
  strsplit(split="") %>% unlist %>% 
  # remove instances of characters you don't care about
  str_remove_all("[,.!'\"]") %>% 
  str_to_lower() %>%
  # make a frequency table of the characters
  table 
letters_space<-c(letters," ")
x<-rep(0,27);names(x)<-letters_space
for (letter in letters_space) x[letter]<-unigrams[letter]
unigrams<-x
barplot(log(unigrams/sum(unigrams,na.rm=TRUE)+1))
```
```{r}
suzanne1<-c(" ",suzanne_eng) %>% paste(collapse="") %>% 
  # tokenize by character (strsplit returns a list, so unlist it)
  strsplit(split="") %>% unlist %>% 
  # remove instances of characters you don't care about
   str_remove_all("[,.!'\"]")  %>% 
   str_to_lower()

suzanne2<-c(suzanne_eng," ") %>% paste(collapse="") %>% 
  # tokenize by character (strsplit returns a list, so unlist it)
  strsplit(split="") %>% unlist %>% 
  # remove instances of characters you don't care about
  str_remove_all("[,.!'\"]")  %>% 
  str_to_lower()

bigrams<-table(suzanne2,suzanne1)

X<-matrix(0,27,27)
row.names(X)<-letters_space
colnames(X)<-letters_space
for (letteri in letters_space)
  for (letterj in letters_space)
    if ((letteri %in% row.names(bigrams))&&(letterj %in% row.names(bigrams))) X[letteri,letterj]<-bigrams[letteri,letterj]

bigrams<-X
```


  

## Document and Links
