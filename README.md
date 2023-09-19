# Self Organized Neural Density Estimation for Complex Time Series Classification


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)  [![made-with-latex](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)



* Petrônio C.  L. Silva  <span itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-1202-2552" href="https://orcid.org/0000-0002-1202-2552" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a></span>
* Omid Orang  <span itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-4077-3775" href="https://orcid.org/0000-0002-4077-3775" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a></span>
* Felipe A. R. da Silva  <span itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0003-4567-8504" href="https://orcid.org/0000-0003-4567-8504" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a></span>
* Fabricio J. E. Costa
* Frederico G. Guimarães <span itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0001-9238-8839" href="https://orcid.org/0000-0001-9238-8839" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a></span>

In case you have any questions, do not hesitate in contact us using the following e-mail: petronio.candido@ifnmg.edu.br


The Time Series Classification is posed such that given a set of $M$ instances of time series $Y = \{ Y_1, \ldots, Y_M \}$, where all instances are IID and have the same number $T$ of samples $y(t) \in \mathbb{R}^N, \; \forall t=1\ldots T$, with $N$ attributes, and a discrete set of labels $C = \{0, \ldots, k\}$, each time series $Y_i$ should be associated with a label $j \in C$.

The present proposal aims to investigate methods for computing both the unconditional probability distribution $P(Y_i)$ and the conditional probability distribution $P(Y_i | c)$, $\forall c \in C$ and $Y_i \in Y$. Representing the data space in terms of the $[0,1]$ probability space is a challenging task due to the high dimensionality of $Y$, with the order of $R^N \times T$ dimensions.

To allow the representation o $Y$ in probability space, the present research proposes an approach based in Neural Networks in two layers: a) the embedding layer implemented using AutoEncoders and b) the density layer implemented using Self Organizing Maps and k-Nearest-Neighbors.

The embedding layer aims to learn a function $f_\mathcal{E}: Y \rightarrow \mathcal{E}$ that maps the original data space $Y \in \mathbb{R}^{T \times N}$ to a smaller dimensional space $\mathcal{E} \in \mathbb{R}^D$ dimensions, where $D \ll T \times N$. The embedding space $\mathcal{E}$ should keep, as much as possible, the properties and topoloty of the original space $Y$ such as the distances $d(Y_i, Y_j)$ of each pair of instances $Y_i, Y_j \in Y$, where $d: Y \times Y \rightarrow \mathbb{R}^+$ is a distance function.

The density layer aims to learn a map $W$ that represent the topology of the embedding space using a square $L \times A$ grid of cells $w_{i,j} \in \mathbb{R}^D$, where each cell $w_{i,j} \in W$ represents a prototype or representative value of a  region of the $\mathcal{E}$ space.

Our proposal learns two more discrete maps, $P$ and $P_k$, where $P[w_{i,j}]$ represents the unconditional probability mass of a the cell $w_{i,j}$ in $\mathcal{E}$ space and $P_k[w_{i,j}]$ represents the the conditional probability mass of a the cell $w_{i,j}$ in $\mathcal{E}$ space given the class label $k \in C$.

With the maps $W$, $P$ and $P_k$ it is possible to calculate the unconditional $P(Y_i)$ and conditional probabilities $P(Y_i|k)$ for any continuous time series in $Y$, by embedding $Y_i$ such that $e_i = f_\mathcal{E}(Y_i)$, finding the k-nearest-neighbors of $e_i$ using the square map $W$ and then interpolate its probabilities weighted by their distances.

