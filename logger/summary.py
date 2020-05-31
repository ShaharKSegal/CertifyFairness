import os

import pandas as pd

import config


def write_experiment_summary(label_num, train_avg_loss, train_acc, train_fair_acc_lst,
                             test_avg_loss, test_acc, test_fair_acc_lst, test_group_acc_lst):
    fair_group_num = len(test_fair_acc_lst)
    group_num = len(test_group_acc_lst)

    def generate_columns(prefixes):
        def generate_accuracy_fair_group_columns(prefix):
            return [f'{prefix}_acc_group{i + 1}' for i in range(fair_group_num)]

        def generate_sampling_weight_columns():
            samp_weight_cols = []
            for group in range(1, fair_group_num + 1):
                for label in range(label_num):
                    samp_weight_cols.append(f'samp_weight_group{group}_label{label}')
            return samp_weight_cols

        def generate_group_columns(prefix):
            group_cols = []
            for group in range(1, fair_group_num + 1):
                for label in range(label_num):
                    group_cols.append(f'{prefix}_acc_group{group}_label{label}')
            return group_cols

        cols = ['run_name', 'dataset_name', 'model_name']
        cols += generate_sampling_weight_columns()
        for prefix in prefixes:
            cols += [f'{prefix}_avg_loss', f'{prefix}_acc']
            cols += generate_accuracy_fair_group_columns(prefix)
        cols += generate_group_columns("test")
        return cols

    def generate_row(train_fair_acc, test_fair_acc, test_group_acc):
        def generate_sampling_weight_data():
            samp_weight_data = [None] * (fair_group_num * label_num)
            if config.args.ignore_weights:
                return samp_weight_data
            for i, group_weight in enumerate(config.args.group_weights):
                samp_weight_data[i] = group_weight
            return samp_weight_data

        if train_fair_acc is None:
            train_fair_acc = [None] * fair_group_num
        elif len(train_fair_acc) < fair_group_num:
            train_fair_acc += [None] * (fair_group_num - len(train_fair_acc))

        data = [config.args.runname, config.args.dataset, config.args.model, *generate_sampling_weight_data(),
                train_avg_loss, train_acc, *train_fair_acc,
                test_avg_loss, test_acc, *test_fair_acc, *test_group_acc]
        return data

    file_path = os.path.join(config.args.save_dir, config.args.summary_file)
    mode = 'a' if os.path.isfile(file_path) else 'w'
    columns = generate_columns(['train', 'test'])
    row = generate_row(train_fair_acc_lst, test_fair_acc_lst, test_group_acc_lst)
    df = pd.DataFrame([row], columns=columns)
    config.log.info(f"Writing summary:\n{df}")
    df.to_csv(file_path, mode=mode, float_format="%.2f", header=mode == 'w', index=False)
