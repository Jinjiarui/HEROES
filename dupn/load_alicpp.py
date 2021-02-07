import numpy as np


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
