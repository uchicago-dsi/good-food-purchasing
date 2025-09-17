# Run a CPU-only scoring engine on your own computer

This is an alternative to [the scoring engine on Google Colab](https://colab.research.google.com/drive/1bFhwSUIzzH_B1gxVqKS_3SYxwxShF4Xe), with the advantage that it locks all software versions, so that it will always work. (Google Colab has updated its Python version and some of its core packages, which broke the scoring engine online.)

**Step 1:** Install Git if you don't have it already. You can find [instructions for your platform here](https://git-scm.com/downloads).

**Step 2:** install Docker if you don't have it already. You can find [instructions for your platform here](https://docs.docker.com/get-started/get-docker/).

**Step 3:** In a terminal, navigate (with `cd`) to a directory where you'd like to put the scoring engine and run

```bash
git clone https://github.com/uchicago-dsi/good-food-purchasing.git
cd good-food-purchasing/scoring-engine

docker build -t good-food-purchasing .
docker run -p 127.0.0.1:8888:8888 good-food-purchasing
```

**Step 4:** When the Docker container is running, Jupyter will provide you with a URL to copy-paste into your browser. It's the last one, the one that contains `http://127.0.0.1:8888/`:

<img width="889" height="614" alt="image" src="https://github.com/user-attachments/assets/5a45d531-ea5c-4de6-8ac3-e46991f30804" />

**Step 5:** Once Jupyter launches, double-click on the `product-group-classifier.ipynb` notebook file in the file browser and then follow the instructions in the notebook.

<img width="379" height="226" alt="image" src="https://github.com/user-attachments/assets/702ee15d-1aa9-42fb-99f3-3431cce96391" />
