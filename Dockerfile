FROM python:3.10.0

WORKDIR /good-food-purchasing

# Install vim as a text editor and build-essential for Make
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install build-essential -y

COPY requirements.pipeline.txt .
COPY pyproject.toml .

# Note: The devcontainer mounts the working directory "over" the copied version
# so copying here is fine. The actual files will be the mounted ones.
COPY . . 

RUN pip install --no-cache-dir -r requirements.dev.txt && \
    pip install --upgrade pip setuptools
RUN pip install -e .

CMD ["/bin/bash"]
