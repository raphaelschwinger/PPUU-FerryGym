FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y python3-opengl xvfb ffmpeg git curl

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# istall poetry 
RUN curl -sSL https://install.python-poetry.org | python3 - 

ENV PATH="/root/.local/bin:$PATH"