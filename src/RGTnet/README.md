[![DOI](https://zenodo.org/badge/380315103.svg)](https://zenodo.org/badge/latestdoi/380315103)
# RgtNet:using synthetic datasets to train an end-to-end CNN for 3-D RGT(Relative Geologic Time) estimation
**This is a [Pytorch](https://pytorch.org/) version of RgtNet for 3-D RGT(Relative Geologic Time) estimation**

## Getting Started with Example Model for RGT estimation

If you would just like to try out a pretrained example model, then you can download the [pretrained model](https://pan.baidu.com/s/1SDTRIc4yggoQYlFPPFa0dQ) [neuc] and use the [demo.ipynb](https://github.com/zfbi/rgtNet/blob/main/demo.ipynb) script to run a demo (example data can be downloaded from [here](https://drive.google.com/drive/folders/12waSUwNHRwdo4g-Ag_xXH1pCNKf2i1Js?usp=sharing)).

### Requirments

```
python>=3.6
torch>=1.0.0
torchvision
torchsummary
natsort
numpy
pillow
plotly
pyparsing
scipy
scikit-image
sklearn
tqdm
```
Install all dependent libraries:
```bash
pip install -r requirements.txt
```

### Dataset

**To train our CNN network, we automatically created 400 pairs of synthetic seismic and corresponding RGT volumes, which were shown 
to be sufficient to train a good RGT estimation network.** 

**The training and validation datasets can be downloaded [here](https://doi.org/10.5281/zenodo.4536561)**

### Training

Run train.sh to start training a new RgtNet model by using the synthetic dataset
```bash
sh train.sh
```

### Validation & Application
Run infer.sh to start applying a new RgtNet model to the synthetic or field seismic data
```bash
sh infer.sh
```

## License

This extension to the Pytorch library is released under a creative commons license which allows for personal and research use only. 
For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
