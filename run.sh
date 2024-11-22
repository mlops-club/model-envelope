#!/bin/bash
PATH=./node_modules/.bin:$PATH

function install {
    npm install
}

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
    
    echo "Dataset installed successfully in ./data directory"
}

function build {
    webpack
}

function start {
    build # Call task dependency
    python -m SimpleHTTPServer 9000
}

function test {
    mocha test/**/*.js
}

function default {
    # Default task to execute
    start
}

function help {
    echo "$0 <task> <args>"
    echo "Tasks:"
    compgen -A function | cat -n
}

TIMEFORMAT="Task completed in %3lR"
time ${@:-default}