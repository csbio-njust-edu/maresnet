import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc


@torch.no_grad()
def eval_training(net, dna_test_loader, softmax_output, args):

    net.eval()
    # test evaluating
    pred_result_test = eval_model(net=net, dataloader=dna_test_loader,
                                  softmax_output=softmax_output,
                                  args=args)
    with open(args.seq_file) as pre_data:
        index_list = []
        seq_list = []
        label_list = []
        lines = pre_data.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            if line.strip()[0] == '>':
                index_list.append(line.strip().split()[0])
            elif line.strip()[0] == 'A' or line.strip()[0] == 'G' or line.strip()[0] == 'C' or line.strip()[0] == 'T' or \
                    line.strip()[0] == 'N':
                seq_list.append(line.strip().split()[0])
        for line in pred_result_test:
            label_list.append(format(line, '.3f'))

        dataframe = pd.DataFrame({'Sequence Name': index_list,
                                  'Sequence': seq_list,
                                  'Possibility of DNA-protein binding': label_list})

        # 创建一个新工作簿并添加一个工作表。
        excel_writer = pd.ExcelWriter(args.output_path + '/result.xlsx', engine='openpyxl')
        dataframe.to_excel(excel_writer, sheet_name='Sheet1', index=False,
                           columns=['Sequence Name', 'Sequence', 'Possibility of DNA-protein binding'],
                           encoding="utf-8", )
        sheet = excel_writer.sheets['Sheet1']
        sheet.column_dimensions['A'].width = 30.0
        sheet.column_dimensions['B'].width = 130.0
        sheet.column_dimensions['C'].width = 38.0
        excel_writer.save()
        excel_writer.close()

    print("prediction result save to file")


def auc_computing(real, pred_numerics):
    for i in range(len(pred_numerics)):
        if np.isnan(pred_numerics[i]):
            pred_numerics[i] = 0.5
    fpr, tpr, thresholds = roc_curve(real, pred_numerics)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def eval_model(net, dataloader, softmax_output, args):
    prob_all = []
    for item in dataloader:
        dna_seqs = item['seq'].to(args.device).float()

        outputs = net(dna_seqs)
        prob = softmax_output(outputs)
        _, pred = outputs.max(1)
        prob_all.extend(prob[:, 1].cpu().numpy())

    return prob_all
