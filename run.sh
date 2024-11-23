#!/bin/bash
PATH=./node_modules/.bin:$PATH

function download-data {
    # Check if kaggle.json exists
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "Error: Kaggle API credentials not found!"
        echo "Please place your kaggle.json file in ~/.kaggle/"
        echo "1. Go to https://www.kaggle.com/settings"
        echo "2. Click 'Create New API Token'"
        echo "3. Move the downloaded kaggle.json to ~/.kaggle/"
        echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi

    # Create data directory if it doesn't exist
    mkdir -p data

    # Download and extract the dataset
    echo "Downloading Runescape Grand Exchange dataset..."
    uvx kaggle datasets download aparoski/runescape-grand-exchange-data -p data
    
    echo "Extracting dataset..."
    cd data && unzip -o runescape-grand-exchange-data.zip && rm runescape-grand-exchange-data.zip
    mv */*.csv ./
    
    echo "Dataset installed successfully in ./data directory"
}

function train-pytorch {
    cd example/train-pytorch
    uv run src/train_pytorch/train.py
}

function test() {
    # Run pytest with coverage
    PYTHONPATH=model-envelope/src pytest \
        model-envelope/tests \
        --cov=model_envelope \
        --cov-report=term-missing \
        -v
}

function help {
    echo "$0 <task> <args>"
    echo "Tasks:"
    compgen -A function | cat -n
}

TIMEFORMAT="Task completed in %3lR"
time ${@:-default}