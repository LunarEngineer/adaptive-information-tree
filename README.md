# Adaptive KD Trees

KD Trees are an extremely efficient method of producing nearest neighbor queries in an orthogonal space.

Embedding vectors are an effective way of mapping *identities* into a space. There is no constraint on the orthogonality of an embedding vector, and we can wave our hands and say 'sure, it can be orthogonal, no biggie'.

If we take a dataset of *educational instructional content* (training samples) and produce a correlated set of embedding vectors over the set of this content we can arbitrarily map these elements into our now *n-dimensional orthogonal space*.

Now, we produce a KD-Tree from the elements of this space and provide subordinate learners with this KD-Tree in order for them to resolve queries of nearest neighbors. (In plain and simple English, give me the most similar content to this *thing*.)

We use *feedback from subordinate learners* to back-propagate *error* and *reward* signals through the embedding vectors, allowing a *reinforcement learning agent* to predict an *adjustment* to *generated back-propagation*, modifying the way the learners move along the gradient.

This ideally results in nearest neighbor queries which are more likely to produce optimal results for queries over a potentially highly dimensional and nonlinear space by mapping the embeddings into a much lower dimensionality orthogonal space.

This embedding space, by virtue of gradient descent, will drift towards a space that is optimal for *cumulative long term reward*.

This package was cloned from a [quite awesome template](https://github.com/AlexIoannides/py-package-template); cheers, I owe you a beer or two!
