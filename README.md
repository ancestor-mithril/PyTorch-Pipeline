# PyTorch-Pipeline

## Simple evironment setup
```
conda create -n 312 -c conda-forge python=3.12
conda activate 312
# conda install pytorch::pytorch torchvision torchaudio -c pytorch  # Mac
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  # Cuda
# conda install pytorch torchvision torchaudio cpuonly -c pytorch  # CPU
conda install tqdm tensorboard
pip install timed-decorator bs-scheduler
```
## How to use
```
git clone https://github.com/ancestor-mithril/PyTorch-Pipeline.git
cd PyTorch-Pipeline
python main.py -h
```


Training on Cifar with ReduceLROnPlateau :
```
python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar10 -data_path ../data -scheduler ReduceLROnPlateau -scheduler_params "{'mode':'min', 'factor':0.5}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half
```

Using a BS Scheduler instead:
```
python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar10 -data_path ../data -scheduler IncreaseBSOnPlateau -scheduler_params "{'mode':'min', 'factor':2.0, 'max_batch_size': 1000}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half
```

## How to do multiple experiments with different parameters

Modify experiment_runner.py and then run it.

## Parameter values

1. `-device`
    * can be `cpu`, `cuda:0`, `mps`
    * default: `cuda:0` 
2. `-learning rate`
    * can be any float
    * default: `0.001`
3. `-bs`
    * can be any uint (limited by RAM/VRAM)
    * default: `64`
4. `-epochs`
    * the maximum number of epochs for training
    * can be any uint
    * default: `200`
5. `-es_patience`
    * if no minimization in train loss is registered for `es_patience` epochs, the training is stopped
    * can be any uint
    * default: `20`
6. `-dataset`
    * implemented datasets: `cifar10`, `cifar100`, `cifar100noisy`, `FashionMNIST`, `MNIST`, `DirtyMNIST`
    * for other datasets: please implement them
    * default: `cifar10`
7. `-data_path`
    * path for dataset files
    * default: `../data`
8. `-scheduler`
    * implemented schedulers: `IncreaseBSOnPlateau`, `ReduceLROnPlateau`, `StepBS`, `StepLR`, `ExponentialBS`, `ExponentialLR`, `PolynomialBS`, `PolynomialLR`, `CosineAnnealingBS`, `CosineAnnealingLR`, `CosineAnnealingBSWithWarmRestarts`, `CosineAnnealingWarmRestarts`, `CyclicBS`, `CyclicLR`, `OneCycleBS`, `OneCycleLR`, `LinearLR`, `LinearBS`
    * for other schedulers: please implement them
    * default: `None`
9. `-scheduler_params`
    * a json string which contains all parameters that need to be passed to the schedulers
    * default: `"{}"`
10. `-model`
    * implemented models: `preresnet18_c10`, `lenet_mnist`, `cnn_mnist`
    * for other models: please implement them
    * default: `preresnet18_c10`
11. `-criterion`
    * implemented criterions: `crossentropy`
    * for other criterions: please implement them
    * default: `crossentropy`
12. `-reduction`
    * implemented reductions: `mean`, `iqr`, `stdmean`
    * for other reductions: please implement them
    * default: `mean`
13. `-loss_scaling`
    * implemented loss scaling: `normal-scaling`
    * for others: please implement them
    * default: `None`
14. `-fill`
    * fill value for transformations
    * can be any float between 0 and 1 (recommended: 0.5)
    * default: `None` (`None` usually means 0, this depends on transformation)
15. `-num_threads`
    * default number of threads to be used by PyTorch on CPU
    * can be any non-zero uint
    * default: `None` (`None` means half of the available threads)
16. `-seed`
    * seed for numpy, torch, cuda and python random generators
    * can be any uint
    * default: `3`
17. `--cutout`
    * if added, uses CutOut (must be implemented first in transformations, currently implemented for `cifar10`, `cifar100`, `cifar100noisy` and `FashionMNIST`)
18. `--autoaug`
    * if added, uses AutoAug (must be implemented first in transformations, currently implemented for `cifar10`, `cifar100`, `cifar100noisy` and `FashionMNIST`)
19. `--tta`
    * if added, does TTA for validation, using original and horizontally flipped tensor (random horizontal flipped should be used during training)
20. `--half`
    * if added, uses torch amp (faster, less VRAM)
21. `--disable_progress_bar`
    * if added, disables tqdm bar which tracks training progression
22. `--verbose`
    * if added, prints stuff to console
23. `--stderr`
    * if added, prints to stderr instead of stdout; works great in combination with silent experiment_runner
   
