# Model Envelope

## Quickstart

```bash
# download the runescape grand exchange CSVs
make download-data

# train a pytorch model
make train-pytorch
```

MLflow now supports tagging models (giving mutable aliases) and is deprecating the old
way of using rigid stages (e.g. `Production`, `Staging`, `Archived`).

`latest` is a reserved tag. I'm not sure how you use it.

To fetch a model using the `mlflow` CLI. You can use

```bash
# fetch the item_1215 model of version 1
mlflow artifacts download \
    --artifact-uri models:/item_1215/1 \
    --dst-path ./model
```

You can also assign the model an alias in the UI by running

```bash
# cd in to the directory where mlruns/ is located
cd ./example/

# run the ui; go to your browser and tag the model, e.g. with "challenger"
mlflow ui --port 5002

# download the model, this time using the aliased version
mlflow artifacts download \
    --artifact-uri models:/item_1215@challenger \
    --dst-path ./model
```

The result of each of these is a file structure like this

```bash
./model
├── artifacts                    # model weights, reports, etc.
│   ├── model_config.json
│   ├── model_state.pt
│   ├── requirements-graph-full.txt
│   ├── requirements-graph-model.txt
│   ├── scaler.pt
│   └── uncommitted_changes.patch
├── code
│   └── train_pytorch            # code needed to run inference
│       ├── __init__.py
│       ├── constants.py
│       ├── dataset.py
│       ├── envelope.py          # load and use the model
│       ├── model.py
│       ├── py.typed
│       └── train.py
├── conda.yaml
├── input_example.json
├── MLmodel
├── python_env.yaml
├── python_model.pkl
├── registered_model_meta
├── requirements.txt             # minimal reqs needed for inference
└── serving_input_example.json
```

**TODO**

Track these aspects of each experimental run:

- [x] Git metadata; capture: (note the solution requires the `git` executable to be available)
    - [x] Git hash
    - [x] Git branch
    - [x] Remote URL
    - [x] Dirty/uncommitted changes (as patch)
    - [ ] Log an HTML rendered diff of that 
- Entrypoint
  - [ ] the script: e.g. `uv run src/train_pytorch/train.py`
  - [ ] CLI args, e.g. `--num-epocs 10`
- System metrics ([docs](https://mlflow.org/docs/latest/system-metrics/index.html))
  - [ ] CPU
  - [ ] Memory
  - [ ] GPU
- Full python deps
  - [ ] Python version
  - [ ] Installed packages
- OS/non-Python deps
  - [ ] Docker base image

Save this metadata with the model:

- [ ] Minimal 3rd Party Python deps
  - [ ] Python version
  - [ ] Installed packages
    - [ ] Use pip-tools or uv to prune subtrees of dependencies, e.g. `jupyterlab`
- [ ] 1st Party Python code
- [ ] Artifacts
- [ ] Pyfunc wrapper
- [ ] Reference to the experiment run