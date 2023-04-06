FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# # install miniconda 

# RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
#     && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
#     && bash /tmp/miniconda.sh -bfp /usr/local \
#     && rm -rf /tmp/miniconda.sh \
#     && conda install -y python=3 \
#     && conda update conda \
#     && apt-get -qq -y remove curl bzip2 \
#     && apt-get -qq -y autoremove \
#     && apt-get autoclean \
#     && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
#     && conda clean --all --yes

# ENV PATH /opt/conda/bin:$PATH

# install additional debian dependencies

RUN apt-get update && apt-get install -y python3-opengl xvfb ffmpeg git

# # Install python 3.9
# ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Europe/Moscow
# RUN apt-get install -y software-properties-common gcc && \
#     add-apt-repository -y ppa:deadsnakes/ppa

# RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install & use pipenv
# COPY Pipfile Pipfile.lock ./
RUN pip install pipenv
# RUN pipenv install 
# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yaml noop.txt /tmp/conda-tmp/
# RUN if [ -f "/tmp/conda-tmp/environment.yaml" ]; then umask 0002 && conda env create --file /tmp/conda-tmp/environment.yaml; fi 

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>tensorflow/tensorflow:latest-gpu
