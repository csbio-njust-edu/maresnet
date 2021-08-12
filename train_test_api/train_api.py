"""
train_api
author Long-Chen Shen
"""
import os
import time


def train(net, dna_training_loader, optimizer, loss_function,
          epoch, args, output_interval,
          is_tensorboard=False, writer=None):

    start = time.time()
    total_loss = 0.0
    correct = 0.0
    net.train()
    for batch_index, item in enumerate(dna_training_loader):

        dna_seqs = item['seq'].to(args.device).float()
        labels = item['label'].to(args.device)

        optimizer.zero_grad()
        outputs = net(dna_seqs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # n_iter = (epoch - 1) * len(dna_training_loader.dataset) + batch_index + 1
        total_loss += loss.item() * dna_seqs.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()

        if batch_index % output_interval == 0 and batch_index != 0:
            current_item = batch_index * args.b + len(dna_seqs)
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]'
                  '\t Average loss: {:0.4f}'
                  '\tAccuracy: {:.4f}'
                  '\tLR: {:0.6f}'.format(
                    total_loss / current_item,
                    correct / current_item,
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=current_item,
                    total_samples=len(dna_training_loader.dataset)
                    ))

    if is_tensorboard:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()
    print('----------- epoch {} training time consumed: {:.2f}s -----------'.format(epoch, finish - start))
    print('Training Epoch: {epoch}\tAverage loss: {:0.4f}\tAccuracy: {:.4f}\tLR: {:0.6f}'.format(
        total_loss / len(dna_training_loader.dataset),
        correct / len(dna_training_loader.dataset),
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
    ))
    log_dic = {
        "epoch": epoch,
        "train_loss": total_loss / len(dna_training_loader.dataset),
        "train_acc": correct / len(dna_training_loader.dataset),
        "lr": optimizer.param_groups[0]['lr'],
    }
    return log_dic
