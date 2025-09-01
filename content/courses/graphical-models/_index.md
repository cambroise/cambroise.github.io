---
# Course title, summary, and position.
linktitle: Graphical Models
summary: Probabilistic frameworks using graphs to represent variables and their conditional dependencies
weight: 1

# Page metadata.
title: Graphical Models
date: "2024-12-16"
lastmod: "2025-09-01"
draft: false  # Is this a draft? true/false
toc: true  # Show table of contents? true/false
type: docs  # Do not modify.

# Add menu entry to sidebar.
menu:
  graphical-models:
    name: Outline
    weight: 1
---

## Outline of the course

1. **Introduction**
   - What is a graphical model?
   - Directed vs. undirected models
   - Uses: parameter reduction, causal exploration
   - Challenges: learning parameters and structure

2. **Directed Graphical Models (DGMs)**
   - Chain rule, conditional independence
   - Examples: Naive Bayes, Markov Chains, Hidden Markov Models
   - Gaussian Bayesian Networks
   - Properties: d-separation, Markov blanket

3. **Learning in DGMs**
   - Parameter learning from complete data
   - Structure learning
     - Chow–Liu algorithm
     - Tree-augmented and mixture models
   - Causal DAGs, interventions, do-calculus

4. **Undirected Graphical Models (UGMs / Markov Random Fields)**
   - Conditional independence via graph separation
   - Hammersley–Clifford theorem
   - Examples: Ising model, Hopfield networks, Boltzmann Machines, RBMs
   - Inference methods: Gibbs sampling, variational approximations

5. **Gaussian Graphical Models (GGMs)**
   - Covariance and precision matrices
   - Conditional independence structure
   - Estimation: covariance selection, Graphical Lasso

6. **Advanced Topics**
   - Inference and Markov properties
   - Relation to exponential family distributions
   - Applications: density estimation, knowledge discovery

7. **Exercises & Projects**
   - Directed/Undirected/Gaussian GM exercises
   - Example projects:
     - MRF simulation
     - Graphical Lasso
     - RBMs on MNIST
     - Structure learning with NoTears or DAG-GNN

---

## Links

- [Lecture notes (pdf)](media/Lecture-graphical-models.pdf)  

---

## Reference document
The lecture closely follows and borrows material from  
**Kevin P. Murphy**, *Machine Learning: A Probabilistic Perspective* (MLAPP), chapters 10, 19, 20, and 26.