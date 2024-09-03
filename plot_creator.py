import glob
import json
import os
import shutil
import warnings
from time import sleep

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

global_stats = {"schedulers": {}, "experiments": {}}


def get_tensorboard_scalars(path):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _ = ea.Reload()
    return {
        scalar: tuple([x.value for x in ea.Scalars(scalar)])
        for scalar in ea.Tags()["scalars"]
    }


def parse_dataset_name(path):
    split = path.split(os.sep)
    assert split[5].startswith("epochs_"), str(split[5:7])
    if split[6].startswith("es_patience_"):
        assert split[8].startswith("scheduler_"), str(split[7:9])
        return split[7]
    else:
        assert split[7].startswith("scheduler_"), str(split[6:8])
        return path.split(os.path.sep)[6]


def parse_scheduler_type(path):
    scheduler = path.split(os.path.sep + "scheduler_")[1].split(os.path.sep)[0]
    scheduler_params = path.split(os.path.sep + "scheduler_params_")[1].split(
        os.path.sep
    )[0]
    bs = path.split(os.path.sep + "bs_")[1].split(os.path.sep)[0]
    return scheduler, scheduler_params, bs


def parse_seed_type(path):
    return path.split(os.path.sep + "seed_")[1].split(os.path.sep)[0]


def get_match_rules():
    match_rules = {
        ("CosineAnnealingBS", "total_iters_100_max_batch_size_256_"): (
            "CosineAnnealingLR",
            "T_max_100_",
        ),
        ("CosineAnnealingBS", "total_iters_200_max_batch_size_256_"): (
            "CosineAnnealingLR",
            "T_max_200_",
        ),
        ("CosineAnnealingBS", "total_iters_50_max_batch_size_256_"): (
            "CosineAnnealingLR",
            "T_max_50_",
        ),
        ("CosineAnnealingBSWithWarmRestarts", "t_0_100_factor_1_max_batch_size_256_"): (
            "CosineAnnealingWarmRestarts",
            "T_0_100_T_mult_1_",
        ),
        ("CosineAnnealingBSWithWarmRestarts", "t_0_100_factor_2_max_batch_size_256_"): (
            "CosineAnnealingWarmRestarts",
            "T_0_100_T_mult_2_",
        ),
        ("CosineAnnealingBSWithWarmRestarts", "t_0_50_factor_1_max_batch_size_256_"): (
            "CosineAnnealingWarmRestarts",
            "T_0_50_T_mult_1_",
        ),
        ("CosineAnnealingBSWithWarmRestarts", "t_0_50_factor_2_max_batch_size_256_"): (
            "CosineAnnealingWarmRestarts",
            "T_0_50_T_mult_2_",
        ),
        ("ExponentialBS", "gamma_1.1_max_batch_size_1000_"): (
            "ExponentialLR",
            "gamma_0.9_",
        ),
        ("ExponentialBS", "gamma_1.01_max_batch_size_1000_"): (
            "ExponentialLR",
            "gamma_0.99_",
        ),
        ("ExponentialBS", "gamma_2_max_batch_size_1000_"): (
            "ExponentialLR",
            "gamma_0.5_",
        ),
        ("IncreaseBSOnPlateau", "mode_min_factor_2.0_max_batch_size_1000_"): (
            "ReduceLROnPlateau",
            "mode_min_factor_0.5_",
        ),
        ("IncreaseBSOnPlateau", "mode_min_factor_5.0_max_batch_size_1000_"): (
            "ReduceLROnPlateau",
            "mode_min_factor_0.2_",
        ),
        ("PolynomialBS", "total_iters_200_max_batch_size_1000_"): (
            "PolynomialLR",
            "total_iters_200_",
        ),
        ("StepBS", "step_size_30_gamma_2.0_max_batch_size_1000_"): (
            "StepLR",
            "step_size_30_gamma_0.5_",
        ),
        ("StepBS", "step_size_50_gamma_2.0_max_batch_size_1000_"): (
            "StepLR",
            "step_size_50_gamma_0.5_",
        ),
        ("StepBS", "step_size_30_gamma_5.0_max_batch_size_1000_"): (
            "StepLR",
            "step_size_30_gamma_0.2_",
        ),
        ("StepBS", "step_size_50_gamma_5.0_max_batch_size_1000_"): (
            "StepLR",
            "step_size_50_gamma_0.2_",
        ),
        (
            "OneCycleBS",
            "total_steps_200_base_batch_size_300_min_batch_size_10_max_batch_size_1000_",
        ): ("OneCycleLR", "total_steps_200_max_lr_0.01_"),
        (
            "CyclicBS",
            "min_batch_size_10_base_batch_size_500_step_size_down_20_mode_triangular2_max_batch_size_1000_",
        ): ("CyclicLR", "base_lr_0.0001_max_lr_0.01_step_size_up_20_mode_triangular2_"),
    }
    match_rules.update({v: k for k, v in match_rules.items()})  # reversed rules
    return match_rules


def transform_scheduler_params(x):
    params = {
        "total_iters_100_max_batch_size_1000_": "100",
        "total_iters_100_max_batch_size_256_": "100",
        "T_max_100_": "100",
        "total_iters_200_max_batch_size_1000_": "200",
        "total_iters_200_max_batch_size_256_": "200",
        "T_max_200_": "200",
        "total_iters_50_max_batch_size_1000_": "50",
        "total_iters_50_max_batch_size_256_": "50",
        "T_max_50_": "50",
        "t_0_100_factor_1_max_batch_size_1000_": "100,1",
        "t_0_100_factor_1_max_batch_size_256_": "100,1",
        "T_0_100_T_mult_1_": "100,1",
        "t_0_100_factor_2_max_batch_size_1000_": "100,2",
        "t_0_100_factor_2_max_batch_size_256_": "100,2",
        "T_0_100_T_mult_2_": "100,2",
        "t_0_50_factor_1_max_batch_size_1000_": "50,1",
        "t_0_50_factor_1_max_batch_size_256_": "50,1",
        "T_0_50_T_mult_1_": "50,1",
        "t_0_50_factor_2_max_batch_size_1000_": "50,2",
        "t_0_50_factor_2_max_batch_size_256_": "50,2",
        "T_0_50_T_mult_2_": "50,2",
        "gamma_1.1_max_batch_size_1000_": "1.1",
        "gamma_0.9_": "0.9",
        "gamma_1.01_max_batch_size_1000_": "1.01",
        "gamma_0.99_": "0.99",
        "gamma_2_max_batch_size_1000_": "2",
        "gamma_0.5_": "0.5",
        "mode_min_factor_2.0_max_batch_size_1000_": "2.0",
        "mode_min_factor_0.5_": "0.5",
        "mode_min_factor_5.0_max_batch_size_1000_": "5.0",
        "mode_min_factor_0.2_": "0.2",
        # 'total_iters_200_max_batch_size_1000_': '200',
        "total_iters_200_": "200",
        "step_size_30_gamma_2.0_max_batch_size_1000_": "30,2.0",
        "step_size_30_gamma_0.5_": "30,0.5",
        "step_size_50_gamma_2.0_max_batch_size_1000_": "50,2.0",
        "step_size_50_gamma_0.5_": "50,0.5",
        "step_size_30_gamma_5.0_max_batch_size_1000_": "30,5.0",
        "step_size_30_gamma_0.2_": "30,0.2",
        "step_size_50_gamma_5.0_max_batch_size_1000_": "50,5.0",
        "step_size_50_gamma_0.2_": "50,0.2",
        # FIXME: tadam
        "total_steps_200_base_batch_size_300_min_batch_size_10_max_batch_size_1000_": "",
        "total_steps_200_max_lr_0.01_": "",
        # FIXME: tadam
        "min_batch_size_10_base_batch_size_500_step_size_down_20_mode_triangular2_max_batch_size_1000_": "",
        "base_lr_0.0001_max_lr_0.01_step_size_up_20_mode_triangular2_": "",
    }
    return params[x]


def skip_runs(tb_paths):
    # Skip criteria
    new_paths = []
    for run in tb_paths:
        scheduler, scheduler_param, initial_batch_size = parse_scheduler_type(run)
        if scheduler_param not in (
            "total_iters_200_max_batch_size_1000_",
            "total_iters_100_max_batch_size_1000_",
            "total_iters_50_max_batch_size_1000_",
            "t_0_100_factor_1_max_batch_size_1000_",
            "t_0_50_factor_1_max_batch_size_1000_",
            "t_0_100_factor_2_max_batch_size_1000_",
            "t_0_50_factor_2_max_batch_size_1000_",
        ):
            new_paths.append(run)

    return new_paths


def match_paths_by_criteria(tb_paths):
    match_rules = get_match_rules()

    groups = []
    while len(tb_paths):
        current_path = tb_paths.pop(0)
        dataset = parse_dataset_name(current_path)
        scheduler, scheduler_param, initial_batch_size = parse_scheduler_type(
            current_path
        )

        seed = parse_seed_type(current_path)

        def is_ok(other_path):
            other_dataset = parse_dataset_name(other_path)
            other_scheduler, other_scheduler_param, other_initial_batch_size = (
                parse_scheduler_type(other_path)
            )
            other_seed = parse_seed_type(other_path)
            if (
                other_dataset != dataset
                or other_initial_batch_size != initial_batch_size
                or other_seed != seed
            ):
                return False
            if match_rules[(scheduler, scheduler_param)] != (
                other_scheduler,
                other_scheduler_param,
            ):
                return False
            return True

        matching = [x for x in tb_paths if is_ok(x)]

        if len(matching) == 0:
            print("WARNING")
            sleep(0.5)
            print("No matching for", current_path)
            sleep(0.5)
            continue
        elif len(matching) != 1:
            print("ERROR")
            sleep(0.5)
            print("ERROR for", scheduler)
            print(current_path)
            for x in matching:
                print(x)
            print()
            sleep(0.5)
            continue

        for x in matching:
            tb_paths.remove(x)

        groups.append((current_path, *matching))

    groups = sorted(groups)
    groups = sorted(groups, key=lambda x: parse_scheduler_type(x[0])[0])
    return groups


def parse_tensorboard(path):
    scalars = get_tensorboard_scalars(path)
    dataset = parse_dataset_name(path)
    scheduler, scheduler_param, initial_batch_size = parse_scheduler_type(path)

    epoch = len(scalars["Train/Time"])
    experiment_time = round(scalars["Train/Time"][epoch - 1] / 3600, 2)
    max_train_accuracy = round(max(scalars["Train/Accuracy"][:epoch]), 2)
    max_val_accuracy = round(max(scalars["Val/Accuracy"][:epoch]), 2)
    initial_learning_rate = round(scalars["Trainer/Learning Rate"][0], 5)
    return {
        "dataset": dataset,
        "scheduler": scheduler,
        "scheduler_param": scheduler_param,
        "experiment_time": experiment_time,
        "experiment_times": scalars["Train/Time"][:epoch],
        "train_epochs": epoch,
        "max_train_accuracy": max_train_accuracy,
        "max_val_accuracy": max_val_accuracy,
        "initial_batch_size": initial_batch_size,
        "initial_learning_rate": initial_learning_rate,
        "train_accuracies": tuple([x / 100 for x in scalars["Train/Accuracy"][:epoch]]),
        "val_accuracies": tuple([x / 100 for x in scalars["Val/Accuracy"][:epoch]]),
        "batch_sizes": scalars["Trainer/Batch Size"][:epoch],
        "learning_rates": scalars["Trainer/Learning Rate"][:epoch],
        "seed": parse_seed_type(path),
        "hash": f"{dataset}_{scheduler}_{scheduler_param}_{initial_batch_size}_{initial_learning_rate}",
    }


def get_scheduler_acronym(x):
    scheduler_acronym = {
        "IncreaseBSOnPlateau": "IBS",
        "ReduceLROnPlateau": "RLR",
        "StepLR": "StepLR",
        "StepBS": "StepBS",
        "CosineAnnealingBS": "CosBS",
        "CosineAnnealingLR": "CosLR",
        "CosineAnnealingBSWithWarmRestarts": "CosWwrBS",
        "CosineAnnealingWarmRestarts": "CosWwrLR",
        "ExponentialBS": "ExpBS",
        "ExponentialLR": "ExpLR",
        "PolynomialBS": "PolyBS",
        "PolynomialLR": "PolyLR",
        "CyclicBS": "CyclicBS",
        "CyclicLR": "CyclicLR",
        "OneCycleBS": "1CycleBS",
        "OneCycleLR": "1CycleLR",
    }
    return scheduler_acronym[x]


def register_stats(group_results):
    r1, r2 = group_results
    length = min([x["train_epochs"] for x in group_results])
    scheduler_combination = f"{r1['dataset']}_{r1['scheduler']}_{r2['scheduler']}"
    experiment_combination = f"{r1['hash']}--__--{r2['hash']}"

    experiment_time_1 = r1["experiment_times"][length - 1] / 3600
    experiment_time_2 = r2["experiment_times"][length - 1] / 3600
    max_val_accuracy_1 = max(r1["val_accuracies"][:length])
    max_val_accuracy_2 = max(r2["val_accuracies"][:length])
    max_train_accuracy_1 = max(r1["train_accuracies"][:length])
    max_train_accuracy_2 = max(r2["train_accuracies"][:length])

    global global_stats
    if scheduler_combination not in global_stats["schedulers"]:
        global_stats["schedulers"][scheduler_combination] = []
    if experiment_combination not in global_stats["experiments"]:
        global_stats["experiments"][experiment_combination] = []

    global_stats["schedulers"][scheduler_combination].append(
        (
            (
                experiment_time_1,
                experiment_time_2,
                max_val_accuracy_1,
                max_val_accuracy_2,
                max_train_accuracy_1,
                max_train_accuracy_2,
            )
        )
    )
    global_stats["experiments"][experiment_combination].append(
        (
            (
                experiment_time_1,
                experiment_time_2,
                max_val_accuracy_1,
                max_val_accuracy_2,
                max_train_accuracy_1,
                max_train_accuracy_2,
            )
        )
    )


def write_stats(summary, tex):
    def gather_stats(data_dict):
        for scheduler in data_dict:
            counts = len(data_dict[scheduler])
            times_1_mean = np.mean([x[0] for x in data_dict[scheduler]])
            times_1_std = np.std([x[0] for x in data_dict[scheduler]])
            times_2_mean = np.mean([x[1] for x in data_dict[scheduler]])
            times_2_std = np.std([x[1] for x in data_dict[scheduler]])

            val_1_mean = np.mean([x[2] for x in data_dict[scheduler]])
            val_1_std = np.std([x[2] for x in data_dict[scheduler]])
            val_2_mean = np.mean([x[3] for x in data_dict[scheduler]])
            val_2_std = np.std([x[3] for x in data_dict[scheduler]])

            train_1_mean = np.mean([x[4] for x in data_dict[scheduler]])
            train_1_std = np.std([x[4] for x in data_dict[scheduler]])
            train_2_mean = np.mean([x[5] for x in data_dict[scheduler]])
            train_2_std = np.std([x[5] for x in data_dict[scheduler]])

            data_dict[scheduler] = [
                (train_1_mean, train_1_std),
                (train_2_mean, train_2_std),
                (val_1_mean, val_1_std),
                (val_2_mean, val_2_std),
                (times_1_mean, times_1_std),
                (times_2_mean, times_2_std),
                {
                    "faster with": (times_2_mean - times_1_mean) / times_2_mean * 100,
                    "accuracy drop": (val_2_mean - val_1_mean) / val_2_mean * 100,
                    "counts": counts,
                },
            ]

    global global_stats
    gather_stats(global_stats["schedulers"])
    gather_stats(global_stats["experiments"])
    with open(summary, "w") as fd:
        json.dump(global_stats, fd, indent=4)
    with open(tex, "w") as fd:
        for scheduler in global_stats["experiments"]:
            (
                (train_1_mean, train_1_std),
                (train_2_mean, train_2_std),
                (val_1_mean, val_1_std),
                (val_2_mean, val_2_std),
                (times_1_mean, times_1_std),
                (times_2_mean, times_2_std),
                *_,
            ) = global_stats["experiments"][scheduler]
            fd.write(
                f'{scheduler.split("--__--")[0]}  '
                f"${train_1_mean:.3f} \\pm {train_1_std:.3f}$ & "
                f"${val_1_mean:.3f} \\pm {val_1_std:.3f}$ & "
                f"${times_1_mean:.3f} \\pm {times_1_std:.3f}$\n"
            )
            fd.write(
                f'{scheduler.split("--__--")[1]}  '
                f"${train_2_mean:.3f} \\pm {train_2_std:.3f}$ & "
                f"${val_2_mean:.3f} \\pm {val_2_std:.3f}$ & "
                f"${times_2_mean:.3f} \\pm {times_2_std:.3f}$\n"
            )


def create_tex(group_results, results_dir):
    tex_file = os.path.join(results_dir, "results_table.txt")

    if not os.path.exists(tex_file):
        open(tex_file, "w").write(
            r"\begin{table}[]"
            "\n"
            r"\resizebox{\textwidth}{!}{"
            "\n"
            r"\begin{tabular}{|c|ccc|cc|c|}"
            "\n"
            r"\hline"
            "\n"
            r"Dataset & Scheduler & First LR & First BS & Train Acc. & Val Acc. & Time (h) \\ \hline"
            "\n"
        )

    length = min([x["train_epochs"] for x in group_results])

    for result in group_results:
        scheduler_name = get_scheduler_acronym(result["scheduler"])
        scheduler_full_name = (
            f"{scheduler_name}({result['scheduler_param'].replace('_', '-')})"
        )
        experiment_time = result["experiment_times"][length - 1] / 3600
        max_train_accuracy = max(result["train_accuracies"][:length])
        max_val_accuracy = max(result["val_accuracies"][:length])
        open(tex_file, "a").write(
            f"{result['dataset']} & "
            f"{scheduler_full_name} & "
            f"{result['initial_learning_rate']} & "
            f"{result['initial_batch_size']} & "
            f"{max_train_accuracy:.2f} & "
            f"{max_val_accuracy:.2f} & "
            f"{experiment_time:.2f}"
            r"\\"
            "\n"
        )

    open(tex_file, "a").write(r"\hline" "\n")


def create_graphics(group_results, results_dir):
    exp_1, exp_2 = group_results
    exp_1["scheduler_param"] = transform_scheduler_params(exp_1["scheduler_param"])
    exp_2["scheduler_param"] = transform_scheduler_params(exp_2["scheduler_param"])

    colors = ["darkred", "royalblue", "orange"]
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.8 / 2))

        length = min(exp_1["train_epochs"], exp_2["train_epochs"])

        def create_df(exp):
            def pad_tuple(x):
                size = length - len(x)
                if size >= 0:
                    return x
                if size > 0:
                    warnings.warn("Problem here")
                    return x + (x[-1],) * size
                return x[:size]

            return pd.DataFrame.from_dict(
                {
                    "epoch": tuple(range(length)),
                    "Train Acc.": pad_tuple(exp["train_accuracies"]),
                    "Test Acc.": pad_tuple(exp["val_accuracies"]),
                    "Batch Size": pad_tuple(exp["batch_sizes"]),
                    "Learning Rate": pad_tuple(exp["learning_rates"]),
                }
            )

        df_1 = create_df(exp_1)
        df_2 = create_df(exp_2)

        # Train
        sns.lineplot(
            x="epoch",
            y="Train Acc.",
            data=df_1,
            linewidth="1",
            color=colors[0],
            linestyle="-",
            alpha=0.7,
            ax=axes[0],
            label=f"{exp_1['scheduler']}({exp_1['scheduler_param']})",
        )
        sns.lineplot(
            x="epoch",
            y="Train Acc.",
            data=df_2,
            linewidth="1",
            color=colors[1],
            linestyle="-",
            alpha=0.7,
            ax=axes[0],
            label=f"{exp_2['scheduler']}({exp_2['scheduler_param']})",
        )
        axes[0].set_ylim(0.0, 1.1)
        axes[0].tick_params(axis="both", which="major", labelsize="x-large")
        axes[0].set_xlabel("epoch", fontsize="xx-large")
        axes[0].set_ylabel("Train Acc.", fontsize="xx-large")
        sns.move_legend(
            axes[0], "upper left", bbox_to_anchor=(0.675, 1.325), fontsize="xx-large"
        )

        # Val
        sns.lineplot(
            x="epoch",
            y="Test Acc.",
            data=df_1,
            linewidth="1",
            color=colors[0],
            linestyle="-",
            alpha=0.7,
            ax=axes[1],
        )
        sns.lineplot(
            x="epoch",
            y="Test Acc.",
            data=df_2,
            linewidth="1",
            color=colors[1],
            linestyle="-",
            alpha=0.7,
            ax=axes[1],
        )
        axes[1].tick_params(axis="both", which="major", labelsize="x-large")
        axes[1].set_xlabel("epoch", fontsize="xx-large")
        axes[1].set_ylabel("Test Acc.", fontsize="xx-large")
        axes[1].set_ylim(0.0, 1.1)

        plt.savefig(
            os.path.join(
                results_dir,
                "plots",
                f"{exp_1['scheduler']}_"
                f"{exp_1['dataset']}_{exp_1['initial_batch_size']}_"
                f"{exp_1['scheduler_param']}_"
                f"{exp_1['initial_learning_rate']}_{exp_1['seed']}_first.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(13, 7.8 / 2), constrained_layout=True)
        # BS
        sns.lineplot(
            x="epoch",
            y="Batch Size",
            data=df_1,
            linewidth="1.5",
            color=colors[0],
            linestyle="-",
            alpha=0.7,
            ax=axes[0],
        )
        sns.lineplot(
            x="epoch",
            y="Batch Size",
            data=df_2,
            linewidth="1.5",
            color=colors[1],
            linestyle="-",
            alpha=0.7,
            ax=axes[0],
        )
        axes[0].tick_params(axis="both", which="major", labelsize="x-large")
        axes[0].set_xlabel("epoch", fontsize="xx-large")
        axes[0].set_ylabel("Batch Size", fontsize="xx-large")

        # LR
        sns.lineplot(
            x="epoch",
            y="Learning Rate",
            data=df_1,
            linewidth="1.5",
            color=colors[0],
            linestyle="-",
            alpha=0.7,
            ax=axes[1],
        )
        sns.lineplot(
            x="epoch",
            y="Learning Rate",
            data=df_2,
            linewidth="1.5",
            color=colors[1],
            linestyle="-",
            alpha=0.7,
            ax=axes[1],
        )
        axes[1].tick_params(axis="both", which="major", labelsize="x-large")
        axes[1].set_xlabel("epoch", fontsize="xx-large")
        axes[1].set_ylabel("Learning Rate", fontsize="xx-large")
        # fig.tight_layout()

        plt.savefig(
            os.path.join(
                results_dir,
                "plots",
                f"{exp_1['scheduler']}_"
                f"{exp_1['dataset']}_{exp_1['initial_batch_size']}_"
                f"{exp_1['scheduler_param']}_"
                f"{exp_1['initial_learning_rate']}_{exp_1['seed']}_second.png",
            ),
            bbox_inches="tight",
        )
        plt.close()


def get_tensorboard_paths(base_dir):
    return glob.glob(f"{base_dir}/**/events.out.tfevents*", recursive=True)


def main(base_dir, results_dir):
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    tex_file = os.path.join(results_dir, "results_table.txt")
    tb_paths = get_tensorboard_paths(base_dir)
    tb_paths = skip_runs(tb_paths)

    if os.path.exists(tex_file):
        os.remove(tex_file)

    for group in tqdm(match_paths_by_criteria(tb_paths)):
        group_results = [parse_tensorboard(x) for x in group]
        create_tex(group_results, results_dir)
        create_graphics(group_results, results_dir)
        register_stats(group_results)

    open(tex_file, "a").write(r"\end{tabular}" "\n" r"}" "\n" r"\end{table}" "\n")

    write_stats(
        os.path.join(results_dir, "summary.json"), os.path.join(results_dir, "tex.txt")
    )


def prepare_for_upload(results_dir):
    os.makedirs("./Upload", exist_ok=True)
    files = [
        "StepBS_cifar100_16_30,2.0_0.001_2525_first.png",
        "StepBS_cifar100_16_50,2.0_0.001_2525_first.png",
        "StepBS_cifar100_16_30,2.0_0.001_2525_second.png",
        "StepBS_cifar100_16_50,2.0_0.001_2525_second.png",
        "IncreaseBSOnPlateau_cifar10_16_2.0_0.001_1_first.png",
        "IncreaseBSOnPlateau_cifar10_16_5.0_0.001_2_first.png",
        "IncreaseBSOnPlateau_cifar10_16_2.0_0.001_1_second.png",
        "IncreaseBSOnPlateau_cifar10_16_5.0_0.001_2_second.png",
        "ExponentialBS_cifar100_16_1.01_0.001_2525_first.png",
        "ExponentialBS_cifar100_16_1.01_0.001_2525_second.png",
        "CosineAnnealingBS_cifar100_32_50_0.001_2_first.png",
        "CosineAnnealingBS_cifar100_32_50_0.001_2_second.png",
        "CosineAnnealingBSWithWarmRestarts_cifar100_32_50,1_0.001_2_first.png",
        "CosineAnnealingBSWithWarmRestarts_cifar100_32_50,1_0.001_2_second.png",
    ]
    for file in files:
        shutil.copy(f"./{results_dir}/plots/{file}", f"./Upload/{file}")
    print("Done")


if __name__ == "__main__":
    main("./all_runs", "Graphics")
    prepare_for_upload("Graphics")
