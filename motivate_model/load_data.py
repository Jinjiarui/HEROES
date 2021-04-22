import numpy as np


def loadCriteoBatch(batchsize, max_seq_len, fin, min_input=3):
    total_data = []
    total_seqlen = []
    total_label = []
    total_click = []
    i = 0
    while i < batchsize:
        tmp_seq = []
        t_label = []
        t_click = []
        try:
            (seq_len, label) = [int(_) for _ in fin.readline().rstrip().split()]
            for j in range(seq_len):
                line = fin.readline().rstrip().split()
                tmp_seq.append([float(line[0])] + [int(_) for _ in line[2:]])
                t_click.append([int(line[1])])
                t_label.append([0])
            t_label[-1] = [label]
            if seq_len < min_input:
                continue
            i += 1
            # 大于最大长度的截断，取后面存在conversion的部分
            tmp_seq = tmp_seq[-max_seq_len:]
            t_click = t_click[-max_seq_len:]
            t_label = t_label[-max_seq_len:]
            # 小于最大长度的补全
            if seq_len < max_seq_len:
                for _ in range(seq_len, max_seq_len):
                    tmp_seq.append([0] * len(tmp_seq[0]))
                    t_label.append([0])
                    t_click.append([0])
            else:
                seq_len = max_seq_len
        except:
            i += 1
            continue
        total_data.append(tmp_seq)
        total_seqlen.append(seq_len)
        total_label.append(t_label)
        total_click.append(t_click)
    return total_data, total_click, total_label, total_seqlen


def loadAliBatch(max_seq_len, fin, batch_seq_list):
    total_data_id = []
    total_data_value = []
    total_seqlen = []
    total_click = []
    total_label = []
    for seq_len in batch_seq_list:
        tmp_id = []
        tmp_value = []
        click = []
        label = []
        try:
            for i in range(seq_len):
                line = fin.readline().rstrip().split()
                splits = [_.split(':') for _ in line[6:]]
                splits = np.reshape(splits, (-1, 3))
                tmp_id.append(splits[:, 1].astype(np.int).tolist())
                tmp_value.append(splits[:, 2].astype(np.float).tolist())
                click.append([int(line[2])])
                label.append([int(line[3])])
            tmp_id = tmp_id[-max_seq_len:]
            tmp_value = tmp_value[-max_seq_len:]
            click = click[-max_seq_len:]
            label = label[-max_seq_len:]
            if seq_len < max_seq_len:
                for _ in range(seq_len, max_seq_len):
                    tmp_id.append([0])
                    tmp_value.append([1])
                    click.append([0])
                    label.append([0])
            else:
                seq_len = max_seq_len
        except:
            continue
        total_data_id += tmp_id
        total_data_value += tmp_value
        total_click.append(click)
        total_label.append(label)
        total_seqlen.append(seq_len)
    return total_data_id, total_data_value, total_click, total_label, total_seqlen


def load_fun(dataset, max_seq_len, fin, batch_seq):
    if dataset == 'Criteo':
        return loadCriteoBatch(batch_seq, max_seq_len, fin)
    elif dataset == 'alicpp':
        return loadAliBatch(max_seq_len, fin, batch_seq)
    else:
        raise ValueError("There is no dataset {} yet! It must be in [Criteo, alicpp]!".format(dataset))


if __name__ == '__main__':
    args = {'dataset': 'alicpp'}
    the_load_fun = load_fun(args)
