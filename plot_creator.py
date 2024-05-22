import glob
import os
import warnings

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm


def get_tensorboard_scalars(path):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _ = ea.Reload()
    return {scalar: tuple([x.value for x in ea.Scalars(scalar)]) for scalar in ea.Tags()["scalars"]}


def parse_dataset_name(path):
    assert path.split(os.path.sep)[5].startswith('epochs_'), str(path.split(os.path.sep)[5:7])
    return path.split(os.path.sep)[6]


def parse_scheduler_type(path):
    scheduler = path.split(os.path.sep + 'scheduler_')[1].split(os.path.sep)[0]
    scheduler_params = path.split(os.path.sep + 'scheduler_params_')[1].split(os.path.sep)[0]
    bs = path.split(os.path.sep + 'bs_')[1].split(os.path.sep)[0]
    return scheduler, scheduler_params, bs


def match_paths_by_criteria(tb_paths):
    match_rules = {
        ('CosineAnnealingBS', 'total_iters_100_max_batch_size_1000_'):
            ('CosineAnnealingLR', 'T_max_100_'),
        ('CosineAnnealingBS', 'total_iters_200_max_batch_size_1000_'):
            ('CosineAnnealingLR', 'T_max_200_'),
        ('CosineAnnealingBS', 'total_iters_50_max_batch_size_1000_'):
            ('CosineAnnealingLR', 'T_max_50_'),

        ('CosineAnnealingBSWithWarmRestarts', 't_0_100_factor_1_max_batch_size_1000_'):
            ('CosineAnnealingWarmRestarts', 'T_0_100_T_mult_1_'),
        ('CosineAnnealingBSWithWarmRestarts', 't_0_100_factor_2_max_batch_size_1000_'):
            ('CosineAnnealingWarmRestarts', 'T_0_100_T_mult_2_'),
        ('CosineAnnealingBSWithWarmRestarts', 't_0_50_factor_1_max_batch_size_1000_'):
            ('CosineAnnealingWarmRestarts', 'T_0_50_T_mult_1_'),
        ('CosineAnnealingBSWithWarmRestarts', 't_0_50_factor_2_max_batch_size_1000_'):
            ('CosineAnnealingWarmRestarts', 'T_0_50_T_mult_2_'),

        ('ExponentialBS', 'gamma_1.1_max_batch_size_1000_'):
            ('ExponentialLR', 'gamma_0.9_'),
        ('ExponentialBS', 'gamma_2_max_batch_size_1000_'):
            ('ExponentialLR', 'gamma_0.5_'),

        ('IncreaseBSOnPlateau', 'mode_min_factor_2.0_max_batch_size_1000_'):
            ('ReduceLROnPlateau', 'mode_min_factor_0.5_'),
        ('IncreaseBSOnPlateau', 'mode_min_factor_5.0_max_batch_size_1000_'):
            ('ReduceLROnPlateau', 'mode_min_factor_0.2_'),

        ('PolynomialBS', 'total_iters_200_max_batch_size_1000_'):
            ('PolynomialLR', 'total_iters_200_'),

        ('StepBS', 'step_size_30_gamma_2.0_max_batch_size_1000_'):
            ('StepLR', 'step_size_30_gamma_0.5_'),
        ('StepBS', 'step_size_50_gamma_2.0_max_batch_size_1000_'):
            ('StepLR', 'step_size_50_gamma_0.5_'),
        ('StepBS', 'step_size_30_gamma_5.0_max_batch_size_1000_'):
            ('StepLR', 'step_size_30_gamma_0.2_'),
        ('StepBS', 'step_size_50_gamma_5.0_max_batch_size_1000_'):
            ('StepLR', 'step_size_50_gamma_0.2_'),

        ('OneCycleBS', 'total_steps_200_base_batch_size_300_min_batch_size_10_max_batch_size_1000_'):
            ('OneCycleLR', 'total_steps_200_max_lr_0.01_'),

        ('CyclicBS', 'min_batch_size_10_base_batch_size_500_step_size_down_20_mode_triangular2_max_batch_size_1000_'):
            ('CyclicLR', 'base_lr_0.0001_max_lr_0.01_step_size_up_20_mode_triangular2_'),
    }
    match_rules.update({v: k for k, v in match_rules.items()})  # reversed rules

    groups = []
    while len(tb_paths):
        current_path = tb_paths.pop(0)
        dataset = parse_dataset_name(current_path)
        scheduler, scheduler_param, initial_batch_size = parse_scheduler_type(current_path)

        def is_ok(other_path):
            other_dataset = parse_dataset_name(other_path)
            other_scheduler, other_scheduler_param, other_initial_batch_size = parse_scheduler_type(other_path)
            if other_dataset != dataset or other_initial_batch_size != initial_batch_size:
                return False
            if match_rules[(scheduler, scheduler_param)] != (other_scheduler, other_scheduler_param):
                return False
            return True

        matching = [x for x in tb_paths if is_ok(x)]

        if len(matching) == 0:
            print("No matching for", current_path)

        for x in matching:
            tb_paths.remove(x)

        groups.append((current_path, *matching))

    return groups


def parse_tensorboard(path):
    scalars = get_tensorboard_scalars(path)
    dataset = parse_dataset_name(path)
    scheduler, scheduler_param, initial_batch_size = parse_scheduler_type(path)

    epoch = len(scalars['Train/Time'])
    experiment_time = round(scalars['Train/Time'][epoch - 1] / 3600, 2)
    max_train_accuracy = round(max(scalars['Train/Accuracy'][:epoch]), 2)
    max_val_accuracy = round(max(scalars['Val/Accuracy'][:epoch]), 2)
    initial_learning_rate = round(scalars['Trainer/Learning Rate'][0], 2)
    return {
        'dataset': dataset,
        'scheduler': scheduler,
        'scheduler_param': scheduler_param,
        'experiment_time': experiment_time,
        'experiment_times': scalars['Train/Time'][:epoch],
        'train_epochs': epoch,
        'max_train_accuracy': max_train_accuracy,
        'max_val_accuracy': max_val_accuracy,
        'initial_batch_size': initial_batch_size,
        'initial_learning_rate': initial_learning_rate,
        'train_accuracies': tuple([x / 100 for x in scalars['Train/Accuracy'][:epoch]]),
        'val_accuracies': tuple([x / 100 for x in scalars['Val/Accuracy'][:epoch]]),
        'batch_sizes': scalars['Trainer/Batch Size'][:epoch],
        'learning_rates': scalars['Trainer/Learning Rate'][:epoch],

    }


def create_tex(group_results, results_dir):
    scheduler_acronym = {
        'IncreaseBSOnPlateau': 'IBS',
        'ReduceLROnPlateau': 'RLR',
        'StepLR': 'StepLR',
        'StepBS': 'StepBS',
        'CosineAnnealingBS': 'CosBS',
        'CosineAnnealingLR': 'CosLR',
        'CosineAnnealingBSWithWarmRestarts': 'CosWwrBS',
        'CosineAnnealingWarmRestarts': 'CosWwrLR',
        'ExponentialBS': 'ExpBS',
        'ExponentialLR': 'ExpLR',
        'PolynomialBS': 'PolyBS',
        'PolynomialLR': 'PolyLR',
    }

    tex_file = os.path.join(results_dir, 'results_table.txt')
    if not os.path.exists(tex_file):
        open(tex_file, 'w').write(
            r'\begin{table}[]''\n'
            r'\resizebox{\textwidth}{!}{''\n'
            r'\begin{tabular}{|c|ccc|cc|c|}''\n'
            r'\hline''\n'
            r'Dataset & Scheduler & First LR & First BS & Train Acc. & Val Acc. & Time (h) \\ \hline''\n'
        )

    length = min([x['train_epochs'] for x in group_results])

    for result in group_results:
        scheduler_name = scheduler_acronym[result['scheduler']]
        experiment_time = round(result['experiment_times'][length - 1] / 3600, 2)
        open(tex_file, 'a').write(
            f"{result['dataset']} & "
            f"{scheduler_name}({result['scheduler_param'].replace('_', ',')}) & "
            f"{result['initial_learning_rate']} & "
            f"{result['initial_batch_size']} & "
            f"{result['max_train_accuracy']} & "
            f"{result['max_val_accuracy']} & "
            f"{experiment_time:.2f}"
            r'\\'
            '\n'
        )
    open(tex_file, 'a').write(r'\hline''\n')


def create_graphics(group_results, results_dir):
    exp_1, exp_2 = group_results

    colors = ['darkred', 'royalblue', 'orange']
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.8 / 2))

        length = min(exp_1['train_epochs'], exp_2['train_epochs'])

        def create_df(exp):
            def pad_tuple(x):
                size = length - len(x)
                if size >= 0:
                    return x
                if size > 0:
                    warnings.warn('Problem here')
                    return x + (x[-1],) * size
                return x[:size]

            return pd.DataFrame.from_dict({
                'epoch': tuple(range(length)),
                'Train Acc.': pad_tuple(exp['train_accuracies']),
                'Val Acc.': pad_tuple(exp['val_accuracies']),
                'Batch Size': pad_tuple(exp['batch_sizes']),
                'Learning Rate': pad_tuple(exp['learning_rates']),
            })

        df_1 = create_df(exp_1)
        df_2 = create_df(exp_2)

        # Train
        sns.lineplot(x="epoch", y="Train Acc.", data=df_1, linewidth='1',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[0],
                     label=f"{exp_1['scheduler']}({exp_1['scheduler_param']})")
        sns.lineplot(x="epoch", y="Train Acc.", data=df_2, linewidth='1',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[0],
                     label=f"{exp_2['scheduler']}({exp_2['scheduler_param']})")
        axes[0].set_ylim(0.0, 1.1)
        sns.move_legend(axes[0], "upper left", bbox_to_anchor=(0.85, 1.2))

        # Val
        sns.lineplot(x="epoch", y="Val Acc.", data=df_1, linewidth='1',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[1])
        sns.lineplot(x="epoch", y="Val Acc.", data=df_2, linewidth='1',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[1])
        axes[1].set_ylim(0.0, 1.1)

        plt.savefig(os.path.join(results_dir, 'plots', f"{exp_1['scheduler']}_"
                                                       f"{exp_1['dataset']}_{exp_1['initial_batch_size']}_"
                                                       f"{exp_1['scheduler_param']}_"
                                                       f"{exp_1['initial_learning_rate']}_first.png"),
                    bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(13, 7.8 / 2))
        # BS
        sns.lineplot(x="epoch", y="Batch Size", data=df_1, linewidth='1.5',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[0])
        sns.lineplot(x="epoch", y="Batch Size", data=df_2, linewidth='1.5',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[0])

        # LR
        sns.lineplot(x="epoch", y="Learning Rate", data=df_1, linewidth='1.5',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[1])
        sns.lineplot(x="epoch", y="Learning Rate", data=df_2, linewidth='1.5',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[1])

        plt.savefig(os.path.join(results_dir, 'plots', f"{exp_1['scheduler']}_"
                                                       f"{exp_1['dataset']}_{exp_1['initial_batch_size']}_"
                                                       f"{exp_1['scheduler_param']}_"
                                                       f"{exp_1['initial_learning_rate']}_second.png"),
                    bbox_inches='tight')
        plt.close()


def get_tensorboard_paths(base_dir):
    return glob.glob(f"{base_dir}/**/events.out.tfevents*", recursive=True)


def main(base_dir, results_dir):
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    tex_file = os.path.join(results_dir, 'results_table.txt')
    tb_paths = get_tensorboard_paths(base_dir)

    if os.path.exists(tex_file):
        os.remove(tex_file)

    for group in tqdm(match_paths_by_criteria(tb_paths)):
        group_results = [parse_tensorboard(x) for x in group]
        create_tex(group_results, results_dir)
        create_graphics(group_results, results_dir)
    open(tex_file, 'a').write(
        r'\end{tabular}''\n'
        r'}''\n'
        r'\end{table}''\n'
    )

    pass


if __name__ == '__main__':
    main('./all_runs', 'Graphics')
