FROM python:3.10.0

WORKDIR /good-food-purchasing

COPY requirements.dev.txt .
COPY pyproject.toml .
# The devcontainer mounts the working directory "over" the copied version
COPY . . 

RUN pip install --no-cache-dir -r requirements.dev.txt && \
    pip install --upgrade pip setuptools
RUN pip install -e .

CMD ["/bin/bash"]
