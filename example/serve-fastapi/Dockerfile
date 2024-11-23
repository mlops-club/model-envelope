# Use Python 3.12 slim image
FROM python:3.12-slim

# Accept build arguments
ARG MODEL_PATH
ARG MODEL_ID

# Set working directory
WORKDIR /app

# copy/create minimum files to be able to install requirements
COPY pyproject.toml README.md src/*/__init__.py ./
RUN mkdir -p ./src/serve_fastapi && touch ./src/serve_fastapi/__init__.py

# Copy model files to a standard location
COPY ${MODEL_PATH} /app/model

# Combine requirements:
# 1. Extract model requirements
RUN if [ -f "/app/model/requirements.txt" ]; then \
        cp /app/model/requirements.txt /app/model-requirements.txt; \
    else \
        touch /app/model-requirements.txt; \
    fi

# 2. Install combined requirements, giving preference to model requirements
RUN pip install --no-cache-dir -r /app/model-requirements.txt && \
    pip install --no-cache-dir --editable ./

COPY ./src ./src

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:///app/model
ENV MODEL_PATH=/app/model

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "serve_fastapi.app:app", "--host", "0.0.0.0", "--port", "8000"] 