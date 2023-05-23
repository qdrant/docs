---
title: Security
weight: 165
---

There are various ways to secure your own Qdrant instance.

For authentication on Qdrant cloud refer to its
[Authentication](https://qdrant.tech/documentation/cloud/cloud-quick-start/#authentication)
section.

## Authentication

*Available as of v1.2.0*

Qdrant supports a simple form of client authentication using a static API key.
This can be used to secure your instance.

To enable API key based authentication in your own Qdrant instance you must
specify a key in the configuration:

```yaml
service:
  # Set an api-key.
  # If set, all requests must include a header with the api-key.
  # example header: `api-key: <API-KEY>`
  #
  # If you enable this you should also enable TLS.
  # (Either above or via an external service like nginx.)
  # Sending an api-key over an unencrypted channel is insecure.
  api_key: your_secret_api_key_here
```

<aside role="alert">TLS must be used to prevent leaking the API key over an unencrypted connection.</aside>

For using API key based authentication in Qdrant cloud see the cloud
[Authentication](https://qdrant.tech/documentation/cloud/cloud-quick-start/#authentication)
section.

The API key then needs to be present in all REST or gRPC requests to your instance.
All official Qdrant clients for Python, Go, and Rust support the API key parameter.

<!---
Examples with clients
-->

```bash
curl \
  -X GET https://localhost:6333 \
  --header 'api-key: your_secret_api_key_here'
```

```python
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://localhost",
    port=6333,
    api_key="your_secret_api_key_here",
)
```
