FROM python:3.10.0

WORKDIR /good-food-purchasing

COPY requirements.dev.txt .
COPY pyproject.toml .
# Note: The devcontainer mounts the working directory "over" the copied version
# so copying here is fine. The actual files will be the mounted ones.
COPY . . 

RUN pip install --no-cache-dir -r requirements.dev.txt && \
    pip install --upgrade pip setuptools
RUN pip install -e .

CMD ["/bin/bash"]
