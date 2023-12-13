## Pretrain vision transformer model for bacterial identification

### Installation

Create conda environment

```
conda env create -f envs/cuda_env.yaml
```

Activate conda environment

```
conda activate cuda_env
```

### Train vision transformer model

```
python src/pretrain.py \
    --configs ${CONFIG_PATH} \
    --secret ${SECRETS_PATH} \
    --test
```

### Extract latent representation from pretrained model

```
python src/extract_features.py \
    ${CHECKPOINT_PATH} \
    ${PRETRAIN_CONFIG_PATH} \
    ${EXTRACT_CONFIG_PATH}
```
