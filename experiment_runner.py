import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from multiprocessing import freeze_support, current_process

import torch.cuda

processes_per_gpu = 1
gpu_count = 1
max_batch_size = 1000

run_index = 0
last_index = -1
if len(sys.argv) >= 2:
    run_index = int(sys.argv[1])
if len(sys.argv) >= 3:
    last_index = int(sys.argv[2])


def run_command(command_idx):
    command, idx = command_idx
    gpu_index = current_process()._identity[0] % gpu_count
    if torch.cuda.is_available():
        command += f' -device cuda:{gpu_index}'
        print("Command:", idx, "on gpu", gpu_index, "on process", current_process()._identity[0])
    else:
        command += ' -device cpu'
        print("Command:", idx, "on cpu on process", current_process()._identity[0])

    today = date.today()
    os.makedirs('./logs', exist_ok=True)
    try:
        start = time.time()
        with open(f"./logs/error_{idx}_{today}.txt", 'a+') as err:
            subprocess.run(command, shell=True, check=True, stderr=err, env={**os.environ, "PYTHONOPTIMIZE": "2"})
        os.remove(f"./logs/error_{idx}_{today}.txt")
        elapsed = (time.time() - start)
        with open("./logs/finished_runs.txt", "a+") as fp:
            fp.write(f"{idx} -> {today} -> " + str(elapsed) + "s + " + command + "\n")
    except subprocess.CalledProcessError:
        with open(f"./logs/failed_runs_{today}.txt", "a+") as fp:
            fp.write(command + '\n')


def create_run(dataset, model, optimizer, seed, epochs, es_patience, batch_size, scheduler_params):
    scheduler_name, scheduler_params = scheduler_params
    scheduler_params = str(scheduler_params).replace(" ", "")
    scheduler_params = str(scheduler_params).replace('"', '\'')
    scheduler_params = '"' + scheduler_params + '"'
    return (
        f" -lr 0.001"
        f" -bs {batch_size}"
        f" -epochs {epochs}"
        f" -es_patience {es_patience}"
        f" -dataset {dataset}"
        f" -data_path ../data"
        f" -scheduler {scheduler_name}"
        f" -scheduler_params {scheduler_params}"
        f" -model {model}"
        f" -seed {seed}"
        f" -fill 0.5"
        f" --cutout"
        f" --autoaug"
        f" --tta"
        f" --disable_progress_bar"
    ) + (" --half" if torch.cuda.is_available() else "")


def generate_runs():
    datasets = [
        # 'cifar10',
        'cifar10', 'cifar100',
        'FashionMNIST'
    ]
    models = [
        'preresnet18_c10'
    ]
    optimizers = [
        'sgd'
    ]
    seeds = [
        2525
    ]
    epochss = [
        200
    ]
    es_patiences = [
        20
    ]
    batch_sizes = [
        16, 32
    ]
    schedulers = [
        # ('IncreaseBSOnPlateau', {'mode': 'min', 'factor': 2.0, 'max_batch_size': max_batch_size}),
        # ('IncreaseBSOnPlateau', {'mode': 'min', 'factor': 5.0, 'max_batch_size': max_batch_size}),
        # ('ReduceLROnPlateau', {'mode': 'min', 'factor': 0.5}),
        # ('ReduceLROnPlateau', {'mode': 'min', 'factor': 0.2}),
        #
        # ('StepBS', {'step_size': 30, 'gamma': 2.0, 'max_batch_size': max_batch_size}),
        # ('StepBS', {'step_size': 50, 'gamma': 2.0, 'max_batch_size': max_batch_size}),
        # ('StepBS', {'step_size': 30, 'gamma': 5.0, 'max_batch_size': max_batch_size}),
        # ('StepBS', {'step_size': 50, 'gamma': 5.0, 'max_batch_size': max_batch_size}),
        #
        # ('StepLR', {'step_size': 30, 'gamma': 0.5}),
        # ('StepLR', {'step_size': 50, 'gamma': 0.5}),
        # ('StepLR', {'step_size': 30, 'gamma': 0.2}),
        # ('StepLR', {'step_size': 50, 'gamma': 0.2}),

        ('ExponentialBS', {'gamma': 1.01, 'max_batch_size': 1000}),
        ('ExponentialLR', {'gamma': 0.99}),

        ('PolynomialBS', {'total_iters':200, 'max_batch_size': 1000}),
        ('PolynomialLR', {'total_iters':200}),

        # ('CosineAnnealingBS', {'total_iters': 200, 'max_batch_size': 1000}),
        # ('CosineAnnealingLR', {'T_max': 200, }),
        #
        # ('CosineAnnealingBSWithWarmRestarts', {'t_0': 100, 'factor': 1, 'max_batch_size': 1000}),
        # ('CosineAnnealingWarmRestarts', {'T_0': 100, 'T_mult': 1}),
        #
        # ('CyclicBS', {'min_batch_size':10, 'base_batch_size': 500, 'step_size_down': 20, 'mode': 'triangular2', 'max_batch_size': 1000}),
        # ('CyclicLR', {'base_lr':0.0001, 'max_lr': 0.01, 'step_size_up': 20, 'mode': 'triangular2'}),
        #
        # ('OneCycleBS', {'total_steps':200, 'base_batch_size': 300, 'min_batch_size': 10, 'max_batch_size': 1000}),
        # ('OneCycleLR', {'total_steps':200, 'max_lr': 0.01}),
    ]

    runs = []
    for dataset, model, optimizer, seed, epochs, es_patience, batch_size, scheduler_params in \
            itertools.product(datasets, models, optimizers, seeds, epochss, es_patiences, batch_sizes, schedulers):
        run = create_run(dataset=dataset, model=model, optimizer=optimizer, seed=seed, epochs=epochs,
                         es_patience=es_patience, batch_size=batch_size, scheduler_params=scheduler_params)
        runs.append(run)

    return [f"python main.py {i}" for i in runs]


if __name__ == "__main__":
    freeze_support()
    runs = generate_runs()

    # # Debug
    # for i in runs:
    #     print(i)

    print(len(runs))
    if last_index == -1 or last_index > len(runs):
        last_index = len(runs)

    with ProcessPoolExecutor(max_workers=gpu_count * processes_per_gpu) as executor:
        executor.map(run_command, [(runs[index], index) for index in range(run_index, last_index)])
