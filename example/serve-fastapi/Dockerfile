# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy model files to a standard location
COPY ./model /app/model
RUN ls -l /app/model
# Install model requirements first, then the application
RUN pip install --no-cache-dir -r /app/model/requirements.txt

COPY pyproject.toml README.md ./
RUN mkdir -p ./src/serve_fastapi && touch ./src/serve_fastapi/__init__.py
RUN pip install --no-cache-dir --editable ./

# Copy application files
COPY ./ ./

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:///app/model
ENV MODEL_PATH=/app/model

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD uvicorn serve_fastapi.app:app --host 0.0.0.0 --port 8000