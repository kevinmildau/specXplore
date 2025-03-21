# For developers: build container using
#   docker build -t specxplore .
# Run container using:
#   docker run -it --rm -v ${PWD}:/specxplore -w /specxplore --network=host specxplore
# connect to the container via vscode remotes, install jupyter extension for demo notebook use. 
# the above run command maps the working directory into the running container, allowing inside development (devcontainer)
# the specXplore code copied during image build is overwritten with these mapped files
# note that introducing breaking changes can lead to docker build problems.

# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    lsb-release \
    gnupg \
    git \
    wget \
    cmake \
    software-properties-common \
    g++

# python3 will be the base python version, python3.8 is the one installed here
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 &&\
    apt-get install -y python3.8-distutils &&\
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

RUN apt-get update && apt-get install -y python3.8-dev
RUN apt-get update && apt-get install -y libhdf5-dev
RUN apt-get update && apt-get install pkg-config -y

# Important for devcontainer file permissions!
ARG USERNAME=containerUser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ENV TZ=Europe/Amsterdam

WORKDIR /specxplore
COPY . /specxplore

RUN pip3.8 install -e .

# Important for devcontainer file permissions!
USER $USERNAME

CMD ["bash"]