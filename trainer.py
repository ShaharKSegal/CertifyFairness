import collections

import config
import dataset

import numpy as np
import pandas as pd
import torch


def train_fairness_network(epoch, model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    correct = 0
    current_samples = 0
    train_len = len(loader.sampler)
    fair_features_count = collections.Counter()
    fair_features_correct = collections.Counter()
    for batch_idx, (data, target, weights, fair_features) in enumerate(loader):
        disp_batch = batch_idx + 1
        if config.args.use_cuda:
            data, target, weights, fair_features = data.cuda(), target.cuda(), weights.cuda(), fair_features.cuda()
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # calculate loss
        loss = criterion(output, target)
        if not config.args.ignore_loss_weights:  # instance weighted loss
            loss = loss * weights
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        current_samples += len(data)
        # correct count per fairness feature
        _update_counters(pred, target, fair_features, fair_features_count, fair_features_correct)
        # log batch
        if disp_batch % config.train_log_interval == 0:
            percent_done = 100. * disp_batch / len(loader)
            config.log.info(f'Train Epoch: {epoch} [{current_samples}/{train_len} ({percent_done:.0f}%)]'
                            f'\tLoss: {loss.item():.6f}')
    avg_loss = epoch_loss / train_len
    acc = 100. * correct / train_len
    config.log.info(f'Train Epoch: {epoch} Average loss: {avg_loss:.4f}\tAccuracy: {correct}/{train_len} ({acc:.0f}%)')

    fair_acc_lst = []
    for fair_label, counter in fair_features_correct.items():
        fair_label_count = fair_features_count[fair_label]
        fair_acc = 100. * counter / fair_label_count
        fair_acc_lst.append(fair_acc)
        config.log.info(f'Train Epoch: {epoch} Fair-label: {fair_label}'
                        f'\tAccuracy: {counter}/{fair_label_count} ({fair_acc:.0f}%)')
    return avg_loss, acc, fair_acc_lst


def evaluate_fairness_network(model, criterion, loader):
    model.eval()
    test_loss = 0
    correct = 0
    fair_features_count = collections.Counter()
    fair_features_correct = collections.Counter()
    group_count = collections.Counter()
    group_correct = collections.Counter()
    likelihood_count = collections.Counter()
    for data, target, weights, fair_features in loader:
        if config.args.use_cuda:
            data, target, fair_features = data.cuda(), target.cuda(), fair_features.cuda()
        output = model(data)
        test_loss += criterion(output, target).mean().item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        # correct count per fairness feature
        _update_counters(pred, target, fair_features, fair_features_count, fair_features_correct, group_count,
                         group_correct, likelihood_count)

    n = len(loader.dataset)
    test_loss /= n
    acc = 100. * correct / n
    config.log.info('Test set: Average loss: {:.4f}\tAccuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, n, acc))
    fair_acc_lst = []
    for fair_label, counter in fair_features_correct.items():
        fair_label_count = fair_features_count[fair_label]
        fair_acc = 100. * counter / fair_label_count
        fair_acc_lst.append(fair_acc)
        config.log.info('Test set: Fair-label: {}\tAccuracy: {}/{} ({:.0f}%)'.format(fair_label,
                                                                                     counter,
                                                                                     fair_label_count,
                                                                                     fair_acc))
    group_acc_lst = []
    for key, counter in group_correct.items():
        count = group_count[key]
        acc = 100. * counter / count
        group_acc_lst.append(acc)
        config.log.info('Test set: fair-label: {} label: {} \tAccuracy: {}/{} ({:.0f}%)'.format(key[0],
                                                                                                key[1],
                                                                                                counter,
                                                                                                count,
                                                                                                acc))
    likelihood_lst = []
    for key, counter in likelihood_count.items():
        fair_label, pred_label = key
        count = fair_features_count[fair_label]
        likelihood = 100. * counter / count
        group_acc_lst.append(likelihood)
        config.log.info('Test set: fair-label: {} pred-label: {} \tLikelihood: {}/{} ({:.0f}%)'.format(fair_label,
                                                                                                       pred_label,
                                                                                                       counter,
                                                                                                       count,
                                                                                                       likelihood))
    return test_loss, acc, fair_acc_lst, group_acc_lst


def evaluate_network_knn(model, train_loader, eval_loader, k):
    model.eval()
    n = len(eval_loader.dataset)
    # generate samples for nearest neighbour
    samples = None
    labels = None
    for data, target, weights, fair_features in train_loader:
        if samples is None:
            samples = data
            labels = target
        else:
            samples = torch.cat((samples, data))
            labels = torch.cat((labels, target))
    if config.args.use_cuda:
        samples, labels = samples.cuda(), labels.cuda()
    cur_idx = 0
    result_columns = ['label', 'model_prediction', 'fairness_group', 'knn_prediction', 'knn_distance']
    result = pd.DataFrame(data=torch.zeros(n, len(result_columns)).numpy(), columns=result_columns)
    for data, target, weights, fair_features in eval_loader:
        num_samps = data.shape[0]
        if config.args.use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        result.loc[cur_idx: cur_idx + num_samps - 1, 'label'] = target.cpu().numpy()
        result.loc[cur_idx: cur_idx + num_samps - 1, 'model_prediction'] = pred.cpu().numpy().flatten()
        result.loc[cur_idx: cur_idx + num_samps - 1, 'fairness_group'] = fair_features.cpu().numpy()
        for i, sample in enumerate(data):
            distances = torch.sum((samples - sample) ** 2, dim=tuple(range(1, samples.ndim)))
            dist, idx = torch.topk(distances, k=k, largest=False)
            knn_pred, _ = torch.mode(labels[idx])
            result.loc[cur_idx + i, 'knn_prediction'] = knn_pred.item()
            result.loc[cur_idx + i, 'knn_distance'] = dist.cpu().numpy().max()
        cur_idx += num_samps
    return result


def sklearn_train_and_evaluate(model, train_dataset: dataset.FairnessDataset, eval_dataset: dataset.FairnessDataset):
    label_col = train_dataset.label_column
    fair_group_col = train_dataset.fairness_groups_column
    fairness_groups = train_dataset.data[fair_group_col].unique()

    X, y = train_dataset.data.drop(label_col, axis=1).values, train_dataset.data[label_col].values
    model.fit(X, y)

    fair_features_count = collections.Counter()
    fair_features_acc = collections.Counter()
    for group in fairness_groups:
        data = eval_dataset.data
        group_data = data[data[fair_group_col] == group]
        X, y = group_data.drop(label_col, axis=1).values, group_data[label_col].values
        fair_features_count[group] = y.shape[0]
        fair_features_acc[group] = model.score(X, y)

    total_count = len(eval_dataset)
    total_acc = sum([fair_features_acc[group] * (count / total_count) for group, count in fair_features_count.items()])
    config.log.info("Eval acc:")
    config.log.info('Test set: Accuracy: {}/{} ({:.0f}%)'.format(int(total_acc * total_count),
                                                                 total_count, 100 * total_acc))
    for fair_label, acc in fair_features_acc.items():
        fair_label_count = fair_features_count[fair_label]
        config.log.info('Test set: Fair-label: {} Accuracy: {}/{} ({:.0f}%)'.format(fair_label,
                                                                                    int(acc * fair_label_count),
                                                                                    fair_label_count,
                                                                                    100 * acc))


def _update_counters(pred, target, fair_features, fair_features_count, fair_features_correct, group_count=None,
                     group_correct=None, likelihood_count=None):
    for fair_label in fair_features.unique():  # add fair group acc and count:
        fair_label = fair_label.item()
        mask = fair_features == fair_label
        label_pred = pred[mask]
        label_target = target[mask]
        fair_features_count[fair_label] += label_pred.shape[0]
        fair_features_correct[fair_label] += label_pred.eq(label_target.view_as(label_pred)).cpu().sum().item()
    for fair_label in fair_features.unique():
        fair_label = fair_label.item()
        if group_count is None:
            continue
        for label in target.unique():  # add group (fair + label) acc and count:
            label = label.item()
            mask = (fair_features == fair_label) & (target == label)
            label_pred = pred[mask]
            label_target = target[mask]
            group_count[(fair_label, label)] += label_pred.shape[0]
            group_correct[(fair_label, label)] += label_pred.eq(label_target.view_as(label_pred)).cpu().sum().item()
        if likelihood_count is None:
            continue
        pred = pred.squeeze()
        for pred_label in pred.unique():  # add (fair) likelihood count:
            pred_label = pred_label.item()
            mask = (fair_features == fair_label) & (pred == pred_label)
            label_pred = pred[mask]
            likelihood_count[(fair_label, pred_label)] += label_pred.shape[0]
