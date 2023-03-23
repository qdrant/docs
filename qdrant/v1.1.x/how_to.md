---
title: How To
weight: 15
---


This sections contains a collection of how-to guides and tutorials for different use cases of Qdrant.


## Optimize qdrant

- different use-cases have different requirements for balancing between memory/speed/precision
- Qdrant is designed to be flexible and customizable, so you can tune it to your needs

![Trafeoff](/docs/tradeoff.png)

### Prefer Low memory footprint with high speed search

- Use quantization with on-disk original vectors
- Optionally disable rescoring
- It will reduce memory footprint and speed up search, but potentially slightly decrease precision.


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



<!--- ## Implement search-as-you-type functionality -->



<!--- ## Move data between clusters -->
