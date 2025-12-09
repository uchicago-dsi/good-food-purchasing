# Use official Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /data

# Install dependencies
RUN pip install --no-cache-dir pandas==2.3.3 openpyxl==3.1.5 requests==2.32.5 tqdm==4.67.1

# Copy your script into the image
COPY p_correct.json /app/
COPY p_subtype_correct.json /app/
COPY predict-names.py /app/

# By default, run the script. Use `exec` so signals are forwarded (good for Ctrl-C).
ENTRYPOINT ["python", "/app/predict-names.py"]
