# Adaptive KD Trees

KD Trees are an extremely efficient method of producing nearest neighbor queries in an orthogonal space.

Embedding vectors are an effective way of mapping *identities* into a space. There is no constaint on the orthogonality of an embedding vector, and we can wave our hands and say 'sure, it can be orthogonal, no biggie'.

If we take a dataset of *educational instructional content* and produce a set of embedding vectors over the set of this content we can arbitrarily map these elements into our now *n-dimensional orthogonal space*.

Now, we produce a KD-Tree from the elements of this space and provide subordinate learners with this KD-Tree in order for them to resolve queries of nearest neighbors.

We use *feedback from the subordinate learners* to back-propagate *error* and *reward* signals through the embedding vectors, allowing a *reinforcement learning agent* to predict an *adjustment* to the *existent back-propagation*, modifying the way the learner moves along the gradient.

This results in nearest neighbor queries which are more likely to produce optimal results for queries mapping over a potentially highly nonlinear surface by mapping the embeddings into a much lower dimensionality orthogonal space.

This embedding space, by virtue of gradient descent, will drift towards a space that is optimal for *cumulative long term reward*.

## Simple Experiments

### Experiment One - Binary Classifiers

Produce a pool of identical binary classifiers.

Test make blobs with various amounts of noise.

### Experiment Two - Classifiers Mixed With Regressors

Produce a pool of various classifiers and regressors.

### Experiment Three - Educational Database and Content Mapping

## Work To Be Done

1. Reinforcement Learning Environment

This is going to be a large chunk of the work, but this needs to be able to do the following.

* Construct a KD Tree (SciPy / SKLearn) from *embedding vectors*.
* Provide nearest neighbors to a given point to a subordinate learner (i.e. when queried for a point, say a misclassified sample, or a high error sample) be able to provide a dataset composed of the nearest neighbors to that point.
* Use a randomly distributed feedback signal from the subordinate learners to conduct an *informed* back-propagation by allowing an agent to select a continuous action, as large as the embedding space, to translate elements of the embedding space (PyTorch.)
* The subordinate learners need to be able to run a training and testing iteration over the data and produce a domain appropriate metric.

2. Application

* Take a dataset and a set of learners and run training cycles.