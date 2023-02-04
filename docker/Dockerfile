## Basis for running detection backends in a container.
##
## Prerequisites:
##   - docker
##   - nvidia driver
##   - nvidia container toolkit
##
## Build with:
##   docker build -t nanogpt:latest .
##
## Save with:
##   docker save  nanogpt:latest  | gzip > nanogpt.tar.gz
##
## Run with:
##   docker run --network host --gpus all --name nanoGPT --rm -it  nanogpt:latest 
## Debug with:
##   docker run --network host --gpus all --name nanoGPT --entrypoint /bin/bash -it  nanogpt:latest 


FROM nvcr.io/nvidia/pytorch:23.01-py3

## Meta information
LABEL cuda.version="12.0" maintainers="Robert Belshaw <rbelshaw@sagarobotics.com>"

WORKDIR /
RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN pip install --upgrade pip wheel setuptools
RUN pip install transformers datasets tiktoken wandb tqdm

# Docker clean-up
RUN rm -rf /var/lib/apt/lists/*

## Create entry point for image
WORKDIR /
# COPY ./entrypoint.sh .
ENTRYPOINT ["/bin/bash"]
CMD ["/bin/bash", "-c"]
