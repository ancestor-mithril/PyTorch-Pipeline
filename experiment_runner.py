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
    command, env, idx = command_idx
    gpu_index = current_process()._identity[0] % gpu_count
    if torch.cuda.is_available():
        command += f" -device cuda:{gpu_index}"
        print(
            "Command:",
            idx,
            "on gpu",
            gpu_index,
            "on process",
            current_process()._identity[0],
        )
    else:
        command += " -device cpu"
        print("Command:", idx, "on cpu on process", current_process()._identity[0])

    env_str = " ".join(f"{k}={v}" for k, v in env.items())
    env.update(os.environ)
    today = date.today()
    os.makedirs("./logs", exist_ok=True)
    try:
        start = time.time()
        file_name = f"./logs/error_{idx}_{today}.txt"
        with open(file_name, "a+") as err:
            subprocess.run(
                command,
                shell=True,
                check=True,
                stderr=err,
                env={**env, "PYTHONOPTIMIZE": "2"},
            )
        elapsed = time.time() - start

        best_score = '0'
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Best: '):
                    best_score = line.split(': ')[1].split(',')[0]

        with open("./logs/finished_runs.txt", "a+") as fp:
            fp.write(f"{idx} -> {today} -> " + str(elapsed) + "s, score: " + best_score + " + " + env_str + " " + command + "\n")
        os.remove(file_name)
    except subprocess.CalledProcessError:
        with open(f"./logs/failed_runs_{today}.txt", "a+") as fp:
            fp.write(env_str + " " + command + "\n")
    except Exception as e:
        import traceback
        traceback.print_exc()


def create_run(
        dataset,
        model,
        optimizer,
        seed,
        epochs,
        es_patience,
        batch_size,
        scheduler_params,
        lr,
        reduction,
        loss_scaling
):
    scheduler_name, scheduler_params = scheduler_params
    scheduler_params = str(scheduler_params).replace(" ", "")
    scheduler_params = str(scheduler_params).replace('"', "'")
    scheduler_params = '"' + scheduler_params + '"'
    return (
        f" -lr {lr}"
        f" -bs {batch_size}"
        f" -epochs {epochs}"
        f" -es_patience {es_patience}"
        f" -dataset {dataset}"
        f" -reduction {reduction}"
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
        f" --stderr"
        f" --verbose"
        f" --cutmix_mixup"
    ) + (
        " --half" if torch.cuda.is_available() else ""
    ) + (
        f" -loss_scaling {loss_scaling}" if loss_scaling is not None else ""
    )


def generate_runs():
    datasets = [
        # 'cifar10',
        # "cifar10",
        "cifar100",
        # "FashionMNIST",
    ]
    models = [
        "preresnet18_c10"
    ]
    optimizers = [
        "sgd"
    ]
    seeds = [
        5
    ]
    epochss = [
        175
    ]
    es_patiences = [
        20
    ]
    batch_sizes = [
        2048
    ]
    lrs = [
        # 0.1, 0.075, 0.05, 0.025, 0.01
        0.175,
        0.15,
        0.125,
        0.1,
    ]
    reductions = [
        "mean"
    ]
    schedulers = [
        # ("StepLR", {"step_size": 20, "gamma": 0.5}),
        ("StepLR", {"step_size": 30, "gamma": 0.5}),
        ("StepLR", {"step_size": 30, "gamma": 0.5}),
        # ("StepLR", {"step_size": 40, "gamma": 0.5}),
    ]
    loss_scalings = [
        None,
        "uniform-scaling", "normal-scaling"
    ]
    loss_scaling_ranges = [
        '0.1', '0.25', '0.5', # '0.75'
    ]

    runs = []
    envs = []
    for (
            dataset,
            model,
            optimizer,
            seed,
            epochs,
            es_patience,
            batch_size,
            scheduler_params,
            lr,
            reduction,
            loss_scaling
    ) in itertools.product(
        datasets,
        models,
        optimizers,
        seeds,
        epochss,
        es_patiences,
        batch_sizes,
        schedulers,
        lrs,
        reductions,
        loss_scalings
    ):
        run = create_run(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            seed=seed,
            epochs=epochs,
            es_patience=es_patience,
            batch_size=batch_size,
            scheduler_params=scheduler_params,
            lr=lr,
            reduction=reduction,
            loss_scaling=loss_scaling
        )
        if loss_scaling is not None:
            for loss_scaling_range in loss_scaling_ranges:
                for patience in ('20', '30'):
                    runs.append(run)
                    envs.append({
                        'loss_scaling_range': loss_scaling_range,
                        'loss_scaling_patience': patience,
                        'NN_norm_type': 'InstanceNorm2d',
                        # 'NN_norm_type': 'Identity',
                    })
        else:
            runs.append(run)
            envs.append({
                'NN_norm_type': 'InstanceNorm2d',
                # 'NN_norm_type': 'Identity',
            })

    return [f"python main.py {i}" for i in runs], envs


if __name__ == "__main__":
    freeze_support()
    runs, envs = generate_runs()

    # # Debug
    # for i, env in zip(runs, envs):
    #     print(env, i)

    print(len(runs))
    if last_index == -1 or last_index > len(runs):
        last_index = len(runs)

    os.makedirs("./logs", exist_ok=True)
    with open("./logs/finished_runs.txt", "a+") as fp:
        fp.write("New experiment: InstanceNorm2d")
        fp.write("\n")

    try:
        with ProcessPoolExecutor(max_workers=gpu_count * processes_per_gpu) as executor:
            executor.map(
                run_command,
                [(runs[index], envs[index], index) for index in range(run_index, last_index)],
            )
    finally:
        with open("./logs/finished_runs.txt", "a+") as fp:
            fp.write("\n")
