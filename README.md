# cytodata_aics

## Getting started

- As you'll suspect from the subsequent bullet points, I recommend you use the internal JupyterHub service I configured at https://jupyterhub.a100.int.allencell.org (if you're not at the Institute, you need to be connected to the VPN)
- **Important**: If you're running from within JupyterHub, precede `pip` commands with `sudo`.
- There is a `requirements.txt` at the root of this repo which contains a list of necessary packages
- To configure access to Minio, the FSSPEC_CONFIG_DIR should be set and pointing to an appropriate config directory.
  - If you're working from within JupyterHub, this is preconfigured for you and you don't need to worry about it
  - If you're working from one of the new compute nodes (`prd-aics-dcp-03x`, where x is 5 through 9), the folder exists (`/etc/fsspec.d`) and
    you simply need to set the environment variable (e.g., do `$ export FSSPEC_CONFIG_DIR=/etc/fsspec.d` before running your jupyter server)
  - Alternatively, you can [check the docs](https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration) for alternative ways of configuring access. Contact Gui, if you need assistance with this.
- For a saner version control experience with notebooks, I propose we use [nbdime](https://nbdime.readthedocs.io/en/latest/).
  - If you're working from within JupyterHub, this should be preconfigured for you and you shouldn't need to worry about it
    - Check it by doing the following:
        - Run `$ which nbdiff` and make sure it outputs `/opt/conda/bin/nbdiff`
        - Run `$ git config --global --get diff.jupyternotebook.command` and make sure it outputs `git-nbdiffdriver diff`
    - If the first command works as expected but the second doesn't, run the following commands:
        - `$ git-nbdiffdriver config --enable --global`
        - `$ git-nbmergedriver config --enable --global`
  - Alternatively, install `nbdime` yourself and configure it as explained above.