"""
test_api
author Long-Chen Shen
"""
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc


@torch.no_grad()
def eval_training(net, dna_valid_loader, dna_test_loader, loss_function, softmax_output,
                  args, epoch=0, df_file=None, log_dic=None, train_after=False):
    print()
    print('============== Evaluating Network Start ==============')
    start = time.time()
    net.eval()
    # valid evaluating
    loss_valid, acc_valid, auc_valid, pred_result_valid = eval_model(net=net, dataloader=dna_valid_loader,
                                                                     loss_function=loss_function,
                                                                     softmax_output=softmax_output,
                                                                     args=args)

    finish = time.time()
    print(' Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        loss_valid,
        acc_valid,
        auc_valid,
        finish - start
    ))

    # test evaluating
    start = time.time()
    loss_test, acc_test, auc_test, pred_result_test = eval_model(net=net, dataloader=dna_test_loader,
                                                                 loss_function=loss_function,
                                                                 softmax_output=softmax_output,
                                                                 args=args)

    finish = time.time()
    cur_result = ' Test set:  Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}, Time consumed:{:.2f}s' \
        .format(epoch, loss_test, acc_test, auc_test, finish - start)
    print(' Test set:  Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        loss_test,
        acc_test,
        auc_test,
        finish - start
    ))
    print('=============== Evaluating Network End ===============')
    print()
    if log_dic is not None and train_after:
        log_dic['valid_loss'] = loss_valid
        log_dic['valid_acc'] = acc_valid
        log_dic['valid_auc'] = auc_valid

        log_dic['test_loss'] = loss_test
        log_dic['test_acc'] = acc_test
        log_dic['test_auc'] = auc_test
        df = pd.read_pickle(df_file)
        df = df.append([log_dic])
        df.reset_index(inplace=True, drop=True)
        df.to_pickle(df_file)

    return auc_valid, cur_result, pred_result_test


def auc_computing(real, pred_numerics):
    for i in range(len(pred_numerics)):
        if np.isnan(pred_numerics[i]):
            pred_numerics[i] = 0.5
    fpr, tpr, thresholds = roc_curve(real, pred_numerics)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def eval_model(net, dataloader, loss_function, softmax_output, args):
    loss_all = 0.0
    correct = 0.0
    prob_all = []
    label_all = []

    for item in dataloader:
        dna_seqs = item['seq'].to(args.device).float()
        labels = item['label'].to(args.device)

        outputs = net(dna_seqs)
        loss = loss_function(outputs, labels)
        prob = softmax_output(outputs)
        loss_all += loss.item() * dna_seqs.size(0)

        _, pred = outputs.max(1)
        prob_all.extend(prob[:, 1].cpu().numpy())
        label_all.extend(labels.cpu().numpy())

        correct += pred.eq(labels).sum().item()
    avg_loss = loss_all / len(dataloader.dataset)
    eval_acc = correct / len(dataloader.dataset)
    eval_auc = auc_computing(label_all, prob_all)
    return avg_loss, eval_acc, eval_auc, prob_all
