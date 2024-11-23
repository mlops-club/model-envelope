#!/bin/bash
set -euo pipefail

# Constants
MODELS_DIR="models"
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"file://${PWD}/../../example/mlruns"}

function pull_model() {
    local model_id=${1:-"latest"}  # Default to "latest"
    local model_path="${MODELS_DIR}/${model_id}"

    if [ -d "$model_path" ]; then
        echo "Model $model_id already exists at $model_path"
        return 0
    fi

    echo "Pulling model $model_id from MLflow..."
    mkdir -p "$model_path"

    if [ "$model_id" = "latest" ]; then
        # Get the latest version of the registered model
        mlflow artifacts download \
            -u "$MLFLOW_TRACKING_URI" \
            --artifact-path "models:/item_1215/latest" \
            -d "$model_path"
    else
        # Pull specific run
        mlflow artifacts download \
            -u "$MLFLOW_TRACKING_URI" \
            -r "$model_id" \
            -d "$model_path"
    fi

    # Extract pip requirements from the model
    if [ -f "${model_path}/requirements.txt" ]; then
        echo "Found model requirements at ${model_path}/requirements.txt"
    else
        echo "No requirements.txt found in model artifacts"
    fi
}

function build_image() {
    local model_id=$1
    local model_path="${MODELS_DIR}/${model_id}"

    # Ensure model exists
    if [ ! -d "$model_path" ]; then
        echo "Model $model_id not found at $model_path"
        echo "Please run 'pull_model $model_id' first"
        return 1
    fi

    # Build the image with model_path as build arg
    docker build \
        --build-arg MODEL_PATH="$model_path" \
        --build-arg MODEL_ID="$model_id" \
        -t "price-predictor:$model_id" \
        .

    echo "Built image price-predictor:$model_id"
}

function serve() {
    local model_id=$1
    local port=${2:-8000}

    # Update docker-compose.yml with the correct image tag
    sed "s|build: .|image: price-predictor:$model_id|g" docker-compose.yml > docker-compose.override.yml

    # Start the service
    docker-compose -f docker-compose.override.yml up -d

    echo "API is running at http://localhost:$port"
    echo "Visit http://localhost:$port/docs for API documentation"
}

function deploy_model() {
    local model_id=$1
    local port=${2:-8000}

    # Pull model if needed
    pull_model "$model_id"

    # Build image
    build_image "$model_id"

    # Serve the model
    serve "$model_id" "$port"
}

function deploy() {
    deploy_model latest
}

function redeploy() {
    rm -rf models/latest
    deploy_model latest
}

function clean() {
    docker-compose down
    docker rmi price-predictor:latest 2>/dev/null || true
    rm -rf models/*
    rm -f docker-compose.override.yml
}

function help() {
    echo "Usage: $0 <command> [args]"
    echo
    echo "Commands:"
    echo "  deploy                            - Deploy latest model version"
    echo "  redeploy                         - Force redeploy of latest model"
    echo "  clean                            - Clean up all artifacts"
    echo "  pull_model <model_id>            - Pull specific model from MLflow"
    echo "  build_image <model_id>           - Build Docker image with model"
    echo "  serve <model_id> [port]          - Serve model API"
    echo "  deploy_model <model_id> [port]   - Pull, build, and serve in one step"
    echo
    echo "Example:"
    echo "  $0 deploy"
    echo "  $0 deploy_model abc123 8000"
}

TIMEFORMAT="Task completed in %3lR"
time "${@:-help}"