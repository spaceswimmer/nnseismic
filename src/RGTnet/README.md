### Environment
To use the code, please use the uv.lock file in the base repository. To create .venv use the following
```bash
uv sync
```
### Training

Run train.sh to start training a new RgtNet model by using the synthetic dataset
```bash
uv run bash train.sh
```

### Validation & Application
Run infer.sh to start applying a new RgtNet model to the synthetic or field seismic data
```bash
uv run bash infer.sh
```


