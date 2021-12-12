# Adaptive Information Trees

Trees are an efficient data structure for representing heirarchical information.

Embedding vectors are an effective way of mapping *identities* into a space. There is no constraint on the orthogonality of the space in which an embedding vector rests in this instance, and we can wave our hands and say 'sure, it can be orthogonal, no biggie'.

If we take a dataset of *educational instructional content* (training samples) and produce a correlated set of embedding vectors over the set of this content we can arbitrarily map these elements into our now *n-dimensional orthogonal space*.

Now, we produce a KD-Tree from the elements of this space and provide subordinate learners with this KD-Tree in order for them to resolve queries of nearest neighbors. (In plain and simple English, give me the most similar content to this *thing*.)

We use *feedback from subordinate learners* to back-propagate *error* and *reward* signals through the embedding vectors, allowing a *reinforcement learning agent* to predict an *adjustment* to *generated back-propagation*, modifying the way the learners move along the gradient.

This ideally results in nearest neighbor queries which are more likely to produce optimal results for queries over a potentially highly dimensional and nonlinear space by mapping the embeddings into a much lower dimensionality orthogonal space.

This embedding space, by virtue of gradient descent, will drift towards a space that is optimal for *cumulative long term reward*.

All of this work requires implementation of a Data Manager class, described below.

This package was cloned from a [quite awesome template](https://github.com/AlexIoannides/py-package-template); cheers, I owe you a beer or two!

## Data Manager Class

The data manager class is a simple concept; when instantiated for the *first* time the data manager is pointed to a dataset that's understandable by PyArrow. That data will be read into one of two data structures: the dataset will be streamed into a single parquet document or a single PyArrow table. That dataset is loaded or unloaded by the Data Manager when directed to do so; loading a Parquet dataset means reading it into a single PyArrow Table, loading an Arrow dataset means memory mapping the file. Unloading the dataset removes all references to the open handle to that file.

When the data is loaded for the first time an `m+t+1` dimensional indexing and embedding vector is created of the same dataset type as the managed data:

* The first element is the *index* within the managed dataset, to allow for efficient joins,
* The adjoining `t` are metadata elements created and updated by the *metadata* function, and
* The remainder are an `m` dimensional embedding vector. The creation of this embedding vector is a choice between a passed set of data, a pseudo-random function, a passed function, or None, in which case a single locational point will be used to describe all data instances.

This embedding metadata is persisted as a memory mapped PyArrow table.

### Data Management for a Dataset

The Data Manager exposes additional, monitored, and validated to function appropriately, API to the Table at this point:

* .dm_get_index(): Return instances of data by index and run the logging function.
* .dm_get_location(): Return instances of data by location and run the logging function. This function can return all instances bound by constraints in addition to servicing nearest neighbor queries by implementing a KD Tree across the embedding space. This will also run the logging function.
* .dm_shard(): Split this dataset via the splitting function into separate Data Managers.

### Data Management for Data Managers

The Data Manager instead manages a dataset of data managers and asynchronously forwards functionality on to those managers, logging that activity, and returning results.

The Data Manager exposes additional, monitored, and validated to function appropriately, API to the Table at this point:

* .dm_collect(): Destroy all subordinate Data Managers and collect their subordinate information into a singular data file prior to reinstantiating the object and updating metadata. This collects to parquet or arrow, dependent on input.

### Metadata Function

This by default tracks only usage frequency statistics, though is overloadable.

#### Logging Function

This by default only updates usage frequency statistics, though is overloadable.
