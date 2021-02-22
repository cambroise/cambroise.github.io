---
date: "2019-05-05T00:00:00+01:00"
draft: true
linktitle: Introduction
menu:
  graphicalmodels:
    parent: Graphical Models
    weight: 1
title: Introduction
toc: true
type: docs
weight: 1
---

# Introduction

## Practical matters

### Reference document
The lecture closely follows and largely borrows material from 
"Machine Learning: A Probabilistic Perspective" from
Kevin P. Murphy, chapters:

  - Chapter 10: Directed graphical models (Bayes nets)
  - Chapter 19:  Undirected graphical models (Markov random fields)
  - Chapter 20:  Exact inference for graphical models
  - Chapter 26: Graphical model structure learning

### Evaluation

The project will be evaluated through of a project 
in R or Python (realized by 2 or 3 student). Each project will be different and rated on the basis of 

  - a code (1/3)
  - a presentation (15 minutes) (1/3)
  - a report (1/3)
  
## Projects list

  - Simulation and estimation of Ising model parameters
  - Programmation of Graphical Lasso
  - Boltzmann Machine
  - Compare Naive Bayes Classifier with  Tree Augmented Bayes Classifier
  - ...


## What is a graphical model ?

A graphical model is a probability distribution in a factorized form

There a two main type of representation of the factorization:

  - directed graphical model
  - undirected graphical model
  
### Why the term graph ?

Conditionnal independences between variables are well modeled via Graphs


## What is it usefull for ?

  - reduce the number of parameters
    - may be used for supervised or unsupervised approaches
  - allow exploratory data analysis by providing a simple graphical representation
    - "approach causality" 

## What problems does it raise ?

  - learning the parameters of a given factorized form
  - learning the structure of the graphical model (factorized form) 



# Directed Graphical Models


## Joint distribution

### Observation
Suppose we observe multiple correlated variables, 
such as words in a document,
pixels in an image, or genes in a microarray.

### Joint distribution
How can we compactly represent the joint distribution $p(x|\theta)$?

## Chain Rule
By the chain rule of probability, 
we can always represent a joint distribution as follows, using
any ordering of the variables:

$$
p(x_{1:V} ) = p(x_1)p(x_2 | x_1) p(x_3|x_2, x_1)p(x_4|x_1, x_2, x_3)... p(x_V |x_{1:V-1})
$$

### The problem of the number of parameters

$O(K) + O(K^2) + O(K^3) +...$
There are $O(K^V)$ parameters in the system


## Conditional independence

The key to efficiently representing large joint distributions is to make some assumptions about conditional independence (CI). 

$$
X \perp Y | Z \Leftrightarrow  p(X, Y |Z) = p(X|Z)p(Y |Z)
$$
$X$ is conditionaly independent of $Y$ knowing Z if once you know $Z$
knowing $Y$ does not help you to guess $X$ 

## Conditional independence: an example

Setting: picking a card at random in a traditional set of cards

  1. if full set of color and values then $color \perp value$
  2. if all diamond faces ($\blacklozenge$) are discarded from the set then 
  $color \perp\!\!\!\!\!\!\diagup value$ but still  $color \perp value | Facecard$ 

$$
P(King | Facecard) = 1/3 = P(\clubsuit | Facecard)
$$
$$
P(King \clubsuit| Facecard) = 1/9 = P(King | Facecard)P(\clubsuit | Facecard)
$$
## Simplification of chain rule

### Simplficiation of  chain rule factorization
Let assume that $x_{t+1} \perp x_{1:t-1} |x_t$, first order Markov assumption. 

$$
p(x_{1:V} ) = p(x_1)\prod_{t-1}^V p(x_t|x_{t-1})
$$
$K-1 + K^2$ parameters

## Graphical models

A graphical model (GM) is a way to represent a joint distribution by making Conditional Independence (CI) assumptions. 

  - the nodes in the graph represent random variables, 
  - and the (lack of) edges represent CI assumptions. 
  
A better name for these models would in fact be ''independence diagrams''

There are several kinds of graphical model, depending on whether 

  - the graph is directed, 
  - undirected, 
  - or some combination of directed and undirected. 

## Example of directed and undirected graphical model

\includegraphics[width=10cm]{Pictures/figure10_1}


## Graph terminology

A graph $G = (V,E)$ consists of 

  - a set of nodes or vertices, $V = \{1,...,V\}$, and 
  - a set of edges, $E = \{(s,t) : s,t \in V\}$. 
  
### Adjacency matrix
We can represent the graph by its adjacency matrix, in which we write $G(s,t) = 1$ to denote $(s,t) \in E$, that is, if $s \rightarrow t$ is an edge in the graph. If $G(s,t) = 1$ iff $G(t,s) = 1$, we say the graph is undirected, otherwise it is directed. 

We usually assume $G(s, s) = 0$, which means there are no self loops.

## Graph terminology

  - **Parent**: For a directed graph, the parents of a node is the set of all nodes that feed into it: $pa(s) \triangleq  \{t : G(t,s) = 1\}$.
  - **Child**: For a directed graph, the children of a node is the set of all nodes that feed out of it: $ch(s) \triangleq \{t : G(s,t) = 1\}$.
  - **Family**: For a directed graph, the family of a node is the node and its parents, $fam(s) = {s} \cup  pa(s)$.
  - **Root**: For a directed graph, a root is a node with no parents.
  - **Leaf**: For a directed graph, a leaf is a node with no children.
  - **Ancestors**: For a directed graph, the ancestors are the parents, grand-parents, etc of a node. That is, the ancestors of t is the set of nodes that connect to t via a trail: $anc(t) \triangleq \{s : s \rightsquigarrow t\}$.
  - **Descendants**: For a directed graph, the descendants are the children, grand-children, etc of a node. That is, the descendants of s is the set of nodes that can be reached via trails from
s: $desc(s)\triangleq \{t:s \rightsquigarrow t\}$.

## Graph terminology

  - **Clique**: For an undirected graph, a clique is a set of nodes that are all neighbors of each other.
  - **A maximal clique** is a clique which cannot be made any larger without losing the clique property.
  - **Neighbors** For any graph, we define the neighbors of a node as the set of all immediately connected nodes, $nbr(s) \triangleq \{t : G(s, t) = 1 v G(t, s) = 1\}$. For an undirected graph, we write $s \sim  t$ to indicate that s and t are neighbors.
  - **Degree**: The degree of a node is the number of neighbors. For directed graphs, we speak of the in-degree and out-degree, which count the number of parents and children.
  - **Cycle or loop**: For any graph, we define a cycle or loop to be a series of nodes such that we can get back to where we started by following edges
  - **DAG** A directed acyclic graph or DAG is a directed graph with no directed cycles.


## Directed graphical models

  - A directed graphical model or DGM is a GM whose graph is a DAG.
  - These are more commonly known as **Bayesian networks**
  - These models are also called **belief networks** 
  - Finally, these models are sometimes called **causal networks**, because the directed arrows are sometimes interpreted as representing causal relations.
  
## Topological ordering of  DAGs

-  nodes can be ordered such that parents come before children
- it can be constructed from any DAG

### The ordered Markov property 

a node only depends on its immediate parents

$$x_s \perp x_{pred(s) \backslash pa(s)} | x_{pa(s)}$$
where pa(s) are the parents of node s, and pred(s) are the predecessors of node s in the ordering. 

## General form of factorization

$$
p(x_{1:V}) = \prod_{t=1}^V p(x_t | x_{pa(t)})
$$
if the Conditional Independence assumptions encoded in DAG G are correct 

# Examples

## Naive Bayes classifiers 



$$
p(y,x) = p(y) \prod_j p(x_j | y)
$$

The naive Bayes assumption is rather naive, since it assumes the features are conditionally independent.


## Markov and hidden Markov models

### Markov chain
$$
p(x_{1:T})=p(x_1)p(x_2 | x_1)p(x_3 | x_2)...=p(x_1) \prod_{t=2}^T p(x_t | x_{t-1})
$$

### Hidden Markov Model
The hidden variables often represent quantities of interest, such as the identity of the word that someone is currently speaking. The observed variables are what we measure, such as the acoustic waveform. 


## Directed Gaussian graphical models

Consider a DGM where all the variables are real-valued, and all the Conditional Proba. Distributions have the following form:
$$
p(x_t | x_{pa(t)} ) = \mathcal N (x_t | \mu_t + \boldsymbol  w_{t}^T x_{pa(t)} , \sigma_t^2 )
$$

### Directed GGM (Gaussian Bayes net)
$$
p(\boldsymbol x) = \mathcal N (\boldsymbol x | \boldsymbol \mu , \boldsymbol \Sigma)
$$



## Directed GGM (Gaussian Bayes net)
For convenience let rewrite the CPDs
$$
x_t = \mu_t + \sum_{s \in pa(t)} w_{ts}(x_s - \mu_s) + \sigma_t z_t
$$
where  $z_t \sim \mathcal N (0, 1)$, $\sigma_t$ is the conditional standard deviation of $x_t$ given its parents, wts is the strength of the $s \rightarrow t$ edge, and $\mu_t$ is the local mean.

### Mean
The global mean is just the concatenation of the local means
$$\mu = (\mu_1, . . . , \mu_D)^t.$$

## Directed GGM (Gaussian Bayes net)

### Covariance matrix
$$
(\boldsymbol x -\boldsymbol \mu) = W(\boldsymbol x - \boldsymbol \mu) + S \boldsymbol z
$$
where $S\triangleq diag(S)$ 
Let consider 
$\boldsymbol e \triangleq S\boldsymbol z = (I-W)(\boldsymbol x -\boldsymbol \mu)$

We have 
$$ \Sigma =cov(\boldsymbol x -\boldsymbol \mu)= cov((I-W)^{-1} \boldsymbol e) = cov(USz)=U S^2 U^t 
$$
where $U = (I-W)^{-1}$

## Examples
Two extreme cases

  - Isolated vertices : Naive Bayes where  $\boldsymbol \Sigma = S$, p vertices, no edges
  - Fully connected Graph: p vertices, $p(p-1)/2$ directed edges


# Learning 

## Learning from complete data (with known graph structure)

If all the variables are fully observed in each case, so there is no missing data and there are no hidden variables, we say the data is complete. 

$$
p(\mathcal D | \boldsymbol \theta) = \prod_{i=1}^N p(x_i|\boldsymbol \theta) =
\prod_{i=1}^N\prod_{t \in V} p(x_{it} | x_{i,pa(t)}, \boldsymbol \theta_t)
$$
The likelihhod decomposes according the **graph structure**

### Discrete distribution

$$
N_{tck}\triangleq \sum_{i=1}^N \mathbb I(x_{i,t} = k, x_{i,pa(t)} = c)
$$
and thus $\hat p(x_{t} = k, x_{pa(t)}=c) = \frac{N_{tck}}{\sum_{k'} N_{tck'}}$
Of course, the MLE suffers from the zero-count 

## Sprinlker Exercice

```{r  echo=FALSE, eval=TRUE, warning=FALSE, message=FALSE }
# plot network function
plot.network <- function(structure, ht = "400px", cols = "darkturquoise", labels = nodes(structure)){
  if(is.null(labels)) labels <- rep("", length(nodes(structure)))
  nodes <- data.frame(id = nodes(structure),
                      label = labels,
                      color = cols,
                      shadow = TRUE
                      )

  edges <- data.frame(from = structure$arcs[,1],
                      to = structure$arcs[,2],
                      arrows = "to",
                      smooth = FALSE,
                      shadow = TRUE,
                      color = "black")

  return(visNetwork(nodes, edges, height = ht, width = "100%"))
}
```

Let us define the structure of the network

```{r  echo=TRUE, eval=TRUE, warning=FALSE, message=FALSE }
library(bnlearn)
library(visNetwork)
variables<-c("Nuageux","Arrosage","Pluie","HerbeMouillee")
net<-empty.graph(variables)
adj = matrix(0L, ncol = 4, nrow = 4, dimnames=list(variables, variables))
adj["Nuageux","Arrosage"]<-1
adj["Nuageux","Pluie"]<-1
adj["Arrosage","HerbeMouillee"]<-1
adj["Pluie","HerbeMouillee"]<-1
amat(net)=adj
```

## Sprinkler Exercice
```{r   echo=TRUE, eval=TRUE, fig.show=TRUE,fig.height=5}
#plot.network(net) # for a nice html plot
plot(net)
```


## Sprinkler Exercice

Simulate a sample according the model


## Exercice Gaussian Bayesian Network 

### Data 
Let consider the following graph
$x_1 \rightarrow x_2 \rightarrow x_3$

$\mathbb E[x_1]=b_1$, $\mathbb E [x_2]=b_2$, $\mathbb E[x_3]=b_3$



### Problem

  - Write the Adajcency matrix with topological ordering
  - Derive the mean vector and covariance matrix of the random vector
  - Simulate Gaussian data
  - Estimtate the parameters from your simulation
  - What improvment could you suggest ?
 
  
# Conditional independence properties of DGMs

## Diverging edges (fork)

With the DAG
$$
A \leftarrow C \rightarrow B
$$
with have 
$$
A \perp\!\!\!\!\!\!\diagup B
$$ but
$$
A \perp B | C
$$

### Exercice 
Show it


## Chain (Head - tail)
With the DAG
$$
A \rightarrow C \rightarrow B
$$
with have 
$$
A \perp\!\!\!\!\!\!\diagup B
$$ but
$$
A \perp B | C
$$
###  Exercice
Show it


## Converging edges (V) and  collider
With the DAG
$$
A \rightarrow C \leftarrow B
$$
with have 
$$
A \perp \diagup B
$$ but
$$
A \perp \!\!\!\!\!\!\diagup B | C
$$

### Exercice 
Show it



### Independence map
 a directed graph $G$ is an I-map (independence map) for p, 
 or that p is Markov wrt G, 
 
  - iff $I(G) \subseteq	 I(p)$, where I(p) is the set of all CI statements that hold for distribution p. 

This allows us to use the graph as a safe proxy for p

### Minimal I-map

  - The fully connected graph is an I-map of all distributions, 
  -  G is a  of Minimal I-map p 
      1. if G is an I-map of p,
      2. if there isno $G' \subseteq G$ which is an I-map of p.

## d-connection:

If G is a directed graph in which X, Y and Z are disjoint sets of vertices, then X and Y are d-connected by Z in G if and only if
there exists an undirected path P between some vertex in X and some vertex in Y such that  

  - for every collider C on P, either C or a descendent of C is in Z, 
  - and no non-collider on P is in Z.

X and Y are d-separated by Z in G if and only if they are not d-connected by Z in G.


## other formulation d-separation definition
 an undirected path P is d-separated by a set of
nodes E iff at least one of the following conditions hold:

  - P contains a chain, $s\rightarrow m\rightarrow t$ or $s\leftarrow m\leftarrow t$ where $m \in E$
  - P contains a fork, $s\leftarrow m\rightarrow t$  where $m \in E$
  - P contains a collider, $s \rightarrow m \leftarrow t$  where 
  $m \notin E$ and nor is any descendant of m.


## d-separation versus conditional independence

a set of nodes A is d-separated from a different set of nodes B given a third observed set E iff each undirected path from every node 
$a \in A$ to every node $b \in B$ is d-separated by E: 


$x_A \perp_G x_B | x_ E \Leftrightarrow$  A is d-separated from B given E 

## Consequences of d-separation

###  directed local Markov property
From the d-separation criterion, one can conclude that
$t \perp nd(t) \backslash pa(t) | pa(t)$
where the non-descendants of a node $nd(t)$ are all the nodes except for its descendants

### ordered Markov property
A special case of directed local Markov property is when we only look at predecessors of a node according to some topological ordering. We have
$t \perp pred(t) \backslash pa(t) | pa(t)$

## Markov blanket

The set of nodes that renders a node t conditionally independent of all the other nodes in the graph is called t’s Markov blanket

$$mb(t)\triangleq pa(t) \cup ch(t) \cup copa(t)$$

The Markov blanket of a node in a DGM is equal to the parents, the children, and the co-parents



# Learning Structure

## Learning tree structures

Since the problem of structure learning for general graphs is NP-hard (Chickering 1996), we start by considering the special case of trees. Trees are special because we can learn their structure efficiently

\includegraphics[width=10cm,height=4cm]{Pictures/tree}

## Joint Distribution associated to a directed tree

A directed tree, with a single root node r, defines a joint distribution as follows

$$
p(x|T) = \prod_{t \in V} p(x_t | x_{pa(t)})
$$
The distribution is a product over the edges and the choice of root does not matter

### Symetrization

To make the model more symmetric, it is preferable to use an undirected tree:

$$
p(x|T) = \prod_{t \in V} p(x_t) \prod_{(s,t) \in E} \frac{p(x_s,x_t)}{p(x_s)p(x_t)}
$$

## Chow-Liu algorithm for finding the ML tree structure (1968)

**Goal**: Chow Liu algorithm constructs  tree distribution approximation that has the minimum Kullback–Leibler divergence to the actual distribution  (that maximizes the data likelihood)

### Principle

  1. Compute weight $I(s,t)$ of each (possible) edge $(s,t)$
  2. Find a maximum weight spanning tree (MST)
  3. Give directions to edges in MST by chosing a root node
 

## Chow-Liu algorithm for finding the ML tree structure (1968)

### log-likelihood

$$
\log P(\boldsymbol \theta | \mathcal D,T) = \sum_{tk} N_{tk} \log p(x_t=k) + \sum_{st} \sum_{jk} N_{stjk} \log \frac{ p (x_s=j,x_t=k)}{ p (x_s=j)  p(x_t=k)}
$$
thus $\hat p(x_t=k) = \frac{N_{tk}}{N}$ and $\hat p (x_s=j,x_t=k) = \frac{N_{stjk}}{N}$.

### Mutual information of a pair of variables
$$
I(s,t)= \sum_{jk} \hat p (x_s=j,x_t=k) \log \frac{\hat p (x_s=j,x_t=k)}{\hat p (x_s=j) \hat p(x_t=k)}
$$

### The Kullback–Leibler divergence 
$$
\frac{\log P(\boldsymbol{ \hat \theta_{ML} }| \mathcal D,T)}{N} = \sum_{tk} \hat p(x_t=k) \log \hat p(x_t=k) + \sum_{st} I(s,t)
$$

## Chow-Liu algorithm 

There are several algorithms for finding a max spanning tree (MST).
The two best known are 
  - Prim’s algorithm and 
  - Kruskal’s algorithm. 
  
Both can be implemented to run in 
$O(E log V )$ time, where $E = V^2$ is the number of edges and $V$ is the number of nodes.

## Exercice Gaussian Chow-Liu

  1. Show that in the Gaussian case, $I(s,t)=-\frac{1}{2} \log(1-\rho^2_{st})$,where $\rho_{st}$ is the correlation coefficient (see Exercise 2.13, Murphy)
  2. Given a realisation of $n$ gaussian vector of size $p$ find the ML tree structured covariance matrix using Chow-Liu algorithm. 

\begin{align}
I(s,t) =& \mathbb E[\log \frac{p(x_s,x_t)}{p(x_s)p(x_t)}]\\
     =& -\frac{1}{2}\log \frac{|\boldsymbol \Sigma|}{|I|}  -\frac{1}{2} \mathbb E[z^t \boldsymbol \Sigma^{-1}z - z^t  \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix} z]\\
=& -\frac{1}{2}\log |\boldsymbol  \Sigma |  =    -\frac{1}{2} \log (1-\rho^2) 
\end{align}
where $z=\begin{bmatrix} x_s \\ x_t\end{bmatrix} - \begin{bmatrix} \mu_s \\ \mu_t\end{bmatrix}$

## TAN: Tree-Augmented Naive Bayes
  - Naive Bayse with Chow-Liu

