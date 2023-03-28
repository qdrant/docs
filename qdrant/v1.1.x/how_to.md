---
title: How To
weight: 15
---


This sections contains a collection of how-to guides and tutorials for different use cases of Qdrant.


## Optimize qdrant

Different use cases have different requirements for balancing between memory, speed, and precision.
Qdrant is designed to be flexible and customizable so you can tune it to your needs.

![Trafeoff](/docs/tradeoff.png)


Let's look deeper into each of those possible optimization scenarios.

### Prefer Low memory footprint with high speed search

The main way to achieve high speed search with low memory footprint is keep vectors on disk at the same time minimizing number of disk reads.

Vector Quantization is one way to achieve this. Quantization converts vectors into a more compact representation, which can be stored in memory and used for search. With smaller vectors you can cache more in RAM and reduce number of disk reads.

To configure in-memory quantization, with on-disk original vectors, you need to create a collection with the following config:

```http
PUT /collections/{collection_name}

{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    },
    "optimizers_config": {
        "memmap_threshold": 20000
    },
    "quantization_config": {
        "scalar": {
            "type": "int8",
            "always_ram": true
        }
    }
}
```

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient("localhost", port=6333)

client.recreate_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        ),
    ),
)
```

`mmmap_threshold` will ensure that vectors will be stored on disk, while `always_ram` will ensure that quantized vectors will be stored in RAM.

Optionally, you can disable rescoring with search `params`, which will reduce disk reads even further, but potentially slightly decrease precision.


```http
POST /collections/{collection_name}/points/search

{
    "params": {
        "quantization": {
            "rescore": false
        }
    },
    "vector": [0.2, 0.1, 0.9, 0.7],
    "limit": 10
}
```

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient("localhost", port=6333)

client.search(
    collection_name="{collection_name}",
    query_vector=[0.2, 0.1, 0.9, 0.7],
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=False
        )
    )
)
```


### Prefer High precision with low memory footprint

- Use on-disk vectors. 
- Disk IOPS and fraction of RAM used for caching will affect search speed significantly
- Precision will not be affected


### Prefer High precision with high speed search

- Use in-memory vectors.
- Use quantization with re-scoring.
- Tune HNSW parameters to achieve desired precision/speed tradeoff

### Latency vs Throughput

- There are two main approaches to measure speed of search: 
  - latency of the request - time from the moment request is submitted to the moment response is received
  - throughput - number of requests per second the system can handle

This approach is not mutually exclusive, but in some cases it might be preferable to optimize for one or another.

To prefer minimizing latency, you can set up qdrant to use as many cores for the single request as possible.
You can do it by setting number of segments in the collection to be equal of a factor of the number of cores in the system. In this case, each segment will be processed in parallel, and the final result will be obtained faster.

To prefer throughput, you can set up qdrant to use as many cores as possible for processing multiple requests in parallel.
To do that, you can configure qdrant to use minimal number of segments, which is usually 2.
Large segments benefit from size of the index and overall smaller number of vector comparisons required to find the nearest neighbors. But at the same time require more time to build index.


## Serve vectors for many independent users

This is a common use case, when you want to serve vector search for multiple independent partitions.
It might a split by users, organizations, or anything else. But for simplicity we will call them users.

Each user should only have access to their own vectors, and should not be able to see vectors of other users.

There are multiple ways to achieve this in Qdrant:

- Use multiple collections, one per user. This approach is the most flexible, but creating lots of collections might have some resource overhead. It is only recommended to split users into multiple collections if you have only a few users and you need to guarantee that each users won't affect each other in any way including performance-wise.


- Use single collection with payload-based partitioning. This approach is more efficient for large number of users, but requires some additional preparations.

In the simple case it is enough to add a `group_id` field to each vector in the collection and use filter along with `group_id` to filter vectors for each user.

<!-- Examples -->

However, indexation speed might become a bottleneck in this case, because each vector of each user will be indexed into the same collection.
It is possible to avoid this bottleneck by skipping building the global vector index for the whole collection and build it for only for individual groups instead.

By doing so, you will be able to index vectors for each user independently, significantly speeding up the process.

One downside of this approach is that global requests (without `group_id` filter) will be slower, because they will require to scan all groups to find the nearest neighbors.

To use this approach, you need to:

- set `payload_m` in hnsw config to some non-zero value. E.g. 16.
- set `m` in hnsw config to 0. This will disable building global index for the whole collection
- Create keyword payload index for `group_id` field.


ToDo: code examples

## Bulk upload a large number of vectors


- If you have enough RAM, you may not care
- If you upload vectors faster than qdrant can index them, you might be out of RAM before you finish uploading
- To prevent this, you can disable indexing during upload
- Also it might help setting mmap threshold lower than indexing threshold


## Choose optimizer parameters

Optimizer is a fundamental architecture component of Qdrant.
It is responsible for indexing, merging, vacuuming, and quantizing segments of the collection.

Optimizer allows to combine dynamic updates of any record in the collection with the ability to perform efficient bulk updates. It is especially important for building efficient indexes, which require knowledge of various statistics and distributions before they can be built.

The parameters, which affect optimizer behavior the most are:

```yaml
# Target amount of segments optimizer will try to keep.
# Real amount of segments may vary depending on multiple parameters:
#  - Amount of stored points
#  - Current write RPS
#
# It is recommended to select default number of segments as a factor of the number of search threads,
# so that each segment would be handled evenly by one of the threads.
# If `default_segment_number = 0`, will be automatically selected by the number of available CPUs
default_segment_number: 0

# Do not create segments larger this size (in KiloBytes).
# Large segments might require disproportionately long indexation times,
# therefore it makes sense to limit the size of segments.
#
# If indexation speed have more priority for your - make this parameter lower.
# If search speed is more important - make this parameter higher.
# Note: 1Kb = 1 vector of size 256
# If not set, will be automatically selected considering the number of available CPUs.
max_segment_size_kb: null

# Maximum size (in KiloBytes) of vectors to store in-memory per segment.
# Segments larger than this threshold will be stored as read-only memmaped file.
# To enable memmap storage, lower the threshold
# Note: 1Kb = 1 vector of size 256
# If not set, mmap will not be used.
memmap_threshold_kb: null

# Maximum size (in KiloBytes) of vectors allowed for plain index.
# Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
# Note: 1Kb = 1 vector of size 256
indexing_threshold_kb: 20000
```

Those parameters are working as a conditional statement, which is evaluated for each segment after each update.
If the condition is true, the segment will be scheduled for optimization.

Values of those parameters will affect how qdrant handles updates of the data.

- If you have enough RAM and CPU, it fine to go with default values - qdrant will index all vectors as fast as possible.
- If you have a limited amount of RAM, you can set `memmap_threshold_kb=20000` same value as `indexing_threshold_kb`. It will ensure that all vectors will be stored on disk as the same optimization iteration as indexation.
- If you are doing bulk updates, you can set `indexing_threshold_kb=100000000` (some very large value) to **disable** indexing during bulk updates. It will speed up the process significantly, but will require additional parameter change after bulk updates are finished.

Depending on your collection, you might have not enough vectors per segment to start building index.
E.g. if you have 100k vecotrs and 8 segments, one for each CPU core, each segment will have only 12.5k vectors, which is not enough to build index.
In this case, you can set `indexing_threshold_kb=5000` to start building index even for small segments.



<!--- ## Implement search-as-you-type functionality -->



<!--- ## Move data between clusters -->
