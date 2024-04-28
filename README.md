# PyTorch-Pipeline

## How to use
```
git clone https://github.com/ancestor-mithril/PyTorch-Pipeline.git
cd PyTorch-Pipeline
PYTHONOPTIMIZE=2 python main.py -device cuda:0 -lr 0.001 -bs 10 -epochs 200 -dataset cifar10 -data_path ../data -scheduler ReduceLROnPlateau -scheduler_params "{'mode':'min', 'factor':0.5}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half
PYTHONOPTIMIZE=2 python main.py -device cuda:0 -lr 0.001 -bs 10 -epochs 200 -dataset cifar10 -data_path ../data -scheduler IncreaseBSOnPlateau -scheduler_params "{'mode':'min', 'factor':2.0, 'max_batch_size': 1000}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta
```
