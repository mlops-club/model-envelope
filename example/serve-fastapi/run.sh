#!/bin/bash
set -e

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Constants
MODELS_DIR="models"
MODEL_NAME="item_1215"
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"file://${THIS_DIR}/../mlruns"}

# echo "Downloading model models:/$MODEL_NAME/1"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} mlflow artifacts download --artifact-uri \"models:/$MODEL_NAME/1\" --dst-path $THIS_DIR/models"
# mlflow artifacts download --artifact-uri "models:/$MODEL_NAME/1" --dst-path $THIS_DIR/models | tr -d '[:space:]'

function pull_model() {
    local version=${1:-"latest"}  # Default to "latest"
    mkdir -p "$THIS_DIR/model"
    mlflow artifacts download --artifact-uri models:/$MODEL_NAME/$version --dst-path $THIS_DIR/model
}

# assumes the model is already downloaded to $THIS_DIR/model
function build() {
    local version=${1:-"latest"}

    # Build the image from THIS_DIR context
    docker build \
        -t "price-predictor:$version" \
        --progress=plain \
        --no-cache \
        "$THIS_DIR"

    # Create docker-compose override file
    sed "s|build: .|image: price-predictor:$version|g" docker-compose.yml > docker-compose.override.yml

    echo "Built image price-predictor:$version"
}

function run() {
    # Start the service
    docker-compose -f docker-compose.override.yml up
}

function clean() {
    # Remove images
    docker rmi price-predictor:latest 2>/dev/null || true
    
    # Clean up files
    rm -rf "${MODELS_DIR}"
    rm -f docker-compose.override.yml
    
    echo "Cleaned up all artifacts"
}

function build_and_run() {
    local version=${1:-"latest"}
    local port=${2:-8000}

    clean
    pull_model "$version"
    build "$version"
    run "$version" "$port"
}

function help() {
    echo "Usage: $0 <command> [args]"
    echo
    echo "Commands:"
    echo "  pull_model [version]            - Pull model from MLflow (defaults to latest)"
    echo "  build [version]           - Build Docker image with model"
    echo "  run [version] [port]            - Run the API service"
    echo "  build_and_run [version] [port]  - Pull, build, and run in one step"
    echo "  clean                            - Clean up all artifacts"
    echo
    echo "Example:"
    echo "  $0 build_and_run latest 8000"
}

TIMEFORMAT="Task completed in %3lR"
time "${@:-help}"