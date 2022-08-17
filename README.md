# cytodata_aics

## Getting started

- There is a `requirements.txt` at the root of this repo which contains a list of necessary packages
- The first cell in this notebook instructs `fsspec` to use `/etc/fsspec.d` as its config folder. This is one way of giving it the credentials to access Minio (an internal S3-like storage, where the data lives)
  However, this folder only exists and contains the appropriate content in the new compute nodes: `prd-aics-dcp-03x`, where x is 5 through 9. By default, `fsspec` looks at `~/.config/fsspec`, so if you're running this
  notebook from a node other than those 5, you can populate that folder instead. Alternatively, you can [check the docs](https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration) for alternative ways of configuring access.