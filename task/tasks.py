import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import config
import dataset
import logger
import model
import trainer


def run_task(run_dir, task_dir, train_loader: torch.utils.data.DataLoader, eval_loader, test_loader):
    if config.args.task in [config.fairness_training, config.evaluate_fairness]:
        net = get_nn_model(run_dir, train_loader.dataset)
        criterion, optimizer = get_criterion_optimizer(net)
        if config.args.task in [config.fairness_training]:
            fairness_training_task(run_dir, train_loader, eval_loader, net, criterion, optimizer)
        elif config.args.task == config.evaluate_fairness:
            trainer.evaluate_fairness_network(net, criterion, eval_loader)
    elif config.args.task == config.sklearn_regular_training:
        sklearn_training_task(train_loader.dataset, eval_loader.dataset)
    elif config.args.task in config.augment_testing_lst:
        net = get_nn_model(run_dir, train_loader.dataset)
        if config.args.task == config.augment_testing_regular:
            augment_testing_task(net, train_loader, eval_loader, test_loader)
        elif config.args.task == config.augment_testing_knn:
            augment_testing_knn_task(task_dir, net, train_loader, eval_loader, test_loader)
        elif config.args.task == config.augment_testing_odin:
            augment_testing_odin_task(task_dir, net, train_loader, eval_loader, test_loader)


def get_nn_model(run_dir, ds):
    config.log.info('==> Building model...')
    if config.args.model not in config.torch_models:
        raise NotImplementedError
    net = None
    if config.args.model == config.Resnet18:
        net = model.resnet18(num_classes=ds.label_count)
    elif config.args.model == config.OverfitResnet:
        net = model.OverfitResNet18(num_classes=ds.label_count)
    elif config.args.model == config.MLP:
        net = model.create_mlp(len(ds.feature_columns), ds.label_count)
    elif config.args.model == config.LeNet:
        if config.args.dataset in config.ColoredMNIST_lst:
            net = model.LeNetMNIST(3, ds.label_count)  # use to colored mnist
        elif config.args.dataset == config.Timit:
            net = model.LeNetTIMIT(1, ds.label_count)  # use for timit
        elif config.args.dataset == config.Timit2Groups:
            # net = model.LeNetTIMIT(1, ds.label_count)  # use for timit
            net = model.LeNetTIMIT2(1, ds.label_count)  # use for timit
    net = net.to(config.args.torch_device)
    if config.args.task in [*config.augment_testing_lst, config.evaluate_fairness]:
        net_path = os.path.join(run_dir, config.args.save_model)
        config.log.info(f'==> Loading model from: {net_path}')
        if config.args.use_cuda:
            state = torch.load(net_path)
        else:
            state = torch.load(net_path, map_location=torch.device('cpu'))
        net.load_state_dict(state['net'])
    # support cuda
    if config.args.use_cuda:
        config.log.info('Using CUDA')
        config.log.info('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
        cudnn.benchmark = True
    return net


def get_criterion_optimizer(net):
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim_cls = optim.Adam
    if config.args.optimizer == config.SGD:
        optim_cls = optim.SGD
    optimizer = optim_cls(net.parameters(), lr=config.args.lr)
    return criterion, optimizer


def fairness_training_task(run_dir, train_loader, eval_loader, net, criterion, optimizer):
    # start training
    stats = []
    train_avg_loss = train_acc = train_fair_acc_lst = None
    test_avg_loss = test_acc = test_fair_acc_lst = test_group_acc_lst = None
    for epoch in range(1, 1 + config.args.max_epochs):
        train_avg_loss, train_acc, train_fair_acc_lst = trainer.train_fairness_network(epoch, net, criterion,
                                                                                       optimizer, train_loader)
        config.log.info("Eval acc:")
        test_avg_loss, test_acc, test_fair_acc_lst, test_group_acc_lst = trainer.evaluate_fairness_network(net,
                                                                                                           criterion,
                                                                                                           eval_loader)
        stats.append([test_avg_loss, test_acc, test_fair_acc_lst, test_group_acc_lst])
        config.log.info('Saving network...')
        state = {'net': net.module.state_dict() if config.args.use_cuda else net.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(run_dir, config.args.save_model))

        config.log.info('Saving test statistics...')
        with open(os.path.join(run_dir, config.args.save_stats), "wb") as f:
            pickle.dump(stats, f)

    # summary file logic
    config.log.info(f'Writing experiment to summary file...')
    logger.write_experiment_summary(train_loader.dataset.label_count, train_avg_loss, train_acc, train_fair_acc_lst,
                                    test_avg_loss, test_acc, test_fair_acc_lst, test_group_acc_lst)


def sklearn_training_task(train_dataset, eval_dataset):
    if not config.args.ignore_weights:
        config.log.warning('ignore_weights option is always on for sklearn models')
    sk_model = None
    if config.args.model == config.RandomForest:
        sk_model = model.random_forest(n_estimators=100)
    elif config.args.model == config.LinearSVM:
        sk_model = model.linear_svm()
    elif config.args.model == config.SVM:
        sk_model = model.svm()
    trainer.sklearn_train_and_evaluate(sk_model, train_dataset, eval_dataset)


def augment_testing_task(net, train_loader, eval_loader, test_loader):
    criterion, optimizer = get_criterion_optimizer(net)
    config.log.info(f'Testing on augmented data')
    trainer.evaluate_fairness_network(net, criterion, eval_loader)
    config.log.info(f'Testing on original data')
    trainer.evaluate_fairness_network(net, criterion, test_loader)


def augment_testing_knn_task(task_dir, net, train_loader, eval_loader, test_loader):
    k = config.args.knn_k
    df: pd.DataFrame = trainer.evaluate_network_knn(net, train_loader, eval_loader, k)
    df_path = os.path.join(task_dir, f'augment_testing_{k}-nn_df.pkl')
    config.log.info(f'Saving {k} nearest neightbour results in DataFrame format to {df_path}')
    df.to_pickle(df_path)
    groups, groups_count = np.unique(df['fairness_group'].values, return_counts=True)
    distances = np.sort(df['knn_distance'].unique())
    accuracy = np.zeros(distances.shape)
    group_accuracy = np.zeros((*distances.shape, *groups.shape))
    for i, distance in enumerate(distances):
        mask = df['knn_distance'] <= distance
        accuracy[i] += (df[mask]['label'] == df[mask]['knn_prediction']).sum()
        accuracy[i] += (df[~mask]['label'] == df[~mask]['model_prediction']).sum()
        for j, group in enumerate(groups):
            group_mask = df['fairness_group'] == group
            group_accuracy[i, j] += (df[mask & group_mask]['label'] == df[mask & group_mask]['knn_prediction']).sum()
            group_accuracy[i, j] += (
                    df[~mask & group_mask]['label'] == df[~mask & group_mask]['model_prediction']).sum()
    plot_path = os.path.join(task_dir, f'augment_testing_{k}-nn_plot_accuracy.png')
    accuracy *= 100 / df.shape[0]
    group_accuracy *= 100 / groups_count

    config.log.info(f'Saving accuracy plot to {plot_path}')
    plt.xlabel(f"{k} nearest neighbours distance threshold")
    plt.ylabel("accuracy")
    plt.plot(distances, accuracy, label='total accuracy')
    for j, group in enumerate(groups):
        plt.plot(distances, group_accuracy[:, j], label=f'group {group} accuracy')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    plot2_path = os.path.join(task_dir, f'augment_testing_{k}-nn_plot_gap.png')
    config.log.info(f'Saving accuracy plot to {plot2_path}')
    plt.xlabel(f"{k} nearest neighbours distance threshold")
    plt.ylabel("fairness gap")
    plt.plot(distances, np.abs(group_accuracy[:, 0] - group_accuracy[:, 1]))
    plt.savefig(plot2_path)
    plt.close()


def augment_testing_odin_task(task_dir, net, train_loader, eval_loader, test_loader):
    temperatures = [1, 10, 100, 1000]
    epsilon = np.linspace(0, 0.01, 30)

    thresholds_dict = {}
    net.eval()
    for temperature in temperatures:
        for loader in [train_loader, eval_loader]:
            out_of_dist = loader != train_loader
            config.log.info(f"Running odin with temperature: {temperature}")
            for data, _, _, _ in loader:
                net.zero_grad()
                if config.args.use_cuda:
                    data = data.cuda()
                grad_outputs = torch.ones(data.shape)
                data.requires_grad = True
                output = net(data)
                output_temper = F.log_softmax(output / temperature)
                loss = output_temper.max(1, keepdim=True)[0]
                torch.autograd.grad(output, data, grad_outputs=grad_outputs)
                # Collect the element-wise sign of the data gradient
                sign_data_grad = data.grad.data.sign()
                sign_data_grad2 = grad_outputs.sign()
                for eps in epsilon:
                    # Create the perturbed image by adjusting each pixel of the input image
                    perturbed_data = data + eps * sign_data_grad
                    # Adding clipping to maintain [0,1] range
                    perturbed_data = torch.clamp(perturbed_data, 0, 1)
                    # detach the data
                    perturbed_data = perturbed_data.detach()
                    pertubed_output = net(perturbed_data)
                    threshold = F.softmax(pertubed_output).max(1, keepdim=False)[0]
                    dict_key = (temperature, eps)
                    if dict_key not in thresholds_dict:
                        thresholds_dict[dict_key] = []
                    thresholds_dict[dict_key].append((threshold.item(), out_of_dist))
            with open(os.path.join(task_dir, 'thresholds_dict.pickle'), 'wb') as handle:
                pickle.dump(thresholds_dict, handle)
