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
  
  - [Bayesian Statistics introduction](media/Appendix-with-notes.pdf)


## Exercices
  - [Exercices about EM algorithm](media/TD_EM.html)
  - [Exercices about Markov Chain](media/TD-markov.html)

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

### Simulation of HMM


```{r}
A<-matrix(c(0.3,0.7,0,0,0.9,0.1,0.6,0,0.4),3,3,byrow = TRUE)
B<-matrix(c(0.5,0.2,0.3,0,0,0,0,0,0,0.2,0.7,0.1,0,0,0,0,0,0.1,0,0.5,0.4),7,3)
X<-c(1,3,4,6)
M<-diag(rep(1,3))-A
M[,3]<-rep(1,3)
Pi<-solve(t(M),b=c(0,0,1))

SimulationHMM<-function(Pi,A,B,n){
  Z<-rep(0,n) # hidden states
  X<-rep(0,n) # emission (obs.)
  K<-length(Pi) # nb of hidden states
  N<-nrow(B)    # nb of modalities
  Z[1]<-sample(1:K,prob = Pi,size = 1,replace=TRUE)
  X[1]<-sample(1:N,prob=B[,Z[1]],size=1)
  for (i in 2:n){
    Z[i]<- sample(1:K,prob=A[Z[i-1],],size=1)
    X[i]<- sample(1:N,prob=B[,Z[i]],size=1)
  }
  return(list(X=X,Z=Z))
}


Hmmsimu<-SimulationHMM(Pi,A,B,100)
plot(Hmmsimu$X,col=Hmmsimu$Z)

SimulationHMMgauss<-function(Pi,A,n){
  Z<-rep(0,n)
  X<-rep(0,n)
  K<-length(Pi)
  N<-nrow(B)
  Z[1]<-sample(1:K,prob = Pi,size = 1,replace=TRUE)
  X[1]<-rnorm(1,mean=2*Z[1])
  for (i in 2:n){
    Z[i]<- sample(1:K,prob=A[Z[i-1],],size=1)
    X[i]<-rnorm(1,mean=2*Z[i])
  }
  return(list(X=X,Z=Z))
}

Hmmsimu<-SimulationHMMgauss(Pi,A,100)
plot(Hmmsimu$X,col=Hmmsimu$Z)
```
## Projet

  - [Projet à rendre pour le 17 décembre 2021](media/Projet-Data-Analysis-2021.pdf)

 - [Projet  à rendre our le 30 janvier 2021](media/Projet-VAE.pdf)


## Document and Links

### Reference books about machine learning

  - [Machine Learning: A Probabilistic Perspective
](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf) from Kevin P. Murphy
  - [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) from  Chris M Bishop 
  

### R base

Official manuals about R base can be retrieved from 

https://cran.r-project.org/manuals.html



Contribution by the community can be retrieved from

https://cran.r-project.org/other-docs.html

The short introduction from Emmanuel Paradis allows a quick start

   - [''R for Beginners''](https://cran.r-project.org/doc/contrib/Paradis-rdebuts_en.pdf) by Emmanuel Paradis.

Longer book allow a deepening. See for example

  - [''Using R for Data Analysis and Graphics - Introduction, Examples and Commentary''](https://cran.r-project.org/doc/contrib/usingR.pdf) by John Maindonald.



### R from RStudio developers

  -  [R for Data Science](https://r4ds.had.co.nz/)   The book of Wickam about more recent R development for data science 

And if you want more see https://www.rstudio.com/resources/books/



