FROM python:3.10-slim

COPY requirements.txt .
COPY notebooks notebooks

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
COPY settings/* /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension

EXPOSE 8888

CMD ["jupyter", "lab", "notebooks/product-group-classifier.ipynb", "--notebook-dir=./notebooks", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
