# Model Envelope

## Quickstart


```bash
# download the runescape grand exchange CSVs
make download-data

# train a pytorch model
make train-pytorch
```

**TODO**

Track these aspects of each experimental run:

- [ ] Git metadata; capture:
    - [ ] Git hash
    - [ ] Git branch
    - [ ] Remote URL
    - [ ] Dirty/uncommitted changes (as patch)
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