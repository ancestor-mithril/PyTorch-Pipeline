# PyTorch-Pipeline

## How to use
```
git clone https://github.com/ancestor-mithril/PyTorch-Pipeline.git
cd PyTorch-Pipeline
PYTHONOPTIMIZE=2 python main.py
```
```
python main.py -scheduler ReduceLROnPlateau -scheduler_params "{'mode':'min', 'factor':0.5}" -fill 0.5 --cutout --autoaug --tta --half
python main.py -scheduler IncreaseBSOnPlateau -scheduler_params "{'mode':'min', 'factor':2.0, 'max_batch_size': 1000}" -fill 0.5 --cutout --autoaug --tta
```
Parameters:
```
device='cuda:0'
lr=0.001
bs=10
epochs=200
dataset='cifar10'
data_path='../data'
scheduler='ReduceLROnPlateau'
scheduler_params="{'mode': 'min', 'factor': 0.5}"
model='preresnet18_c10'
fill=0.5
num_threads=6
seed=3
cutout=True
autoaug=True
tta=True
half=True
disable_progress_bar=False
```
