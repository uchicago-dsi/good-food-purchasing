# Use official Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /data

# Install dependencies
RUN pip install --no-cache-dir numpy==2.3.5 pandas==2.3.3 openpyxl==3.1.5 requests==2.32.5

# Copy script into the image
COPY category-instructions.md /app/
COPY category-schema.json /app/
COPY tag-instructions.md /app/
COPY tag-schema.json /app/
COPY predict-names.py /app/

# By default, run the script. Use `exec` so signals are forwarded (good for Ctrl-C).
ENTRYPOINT ["python", "/app/predict-names.py"]
