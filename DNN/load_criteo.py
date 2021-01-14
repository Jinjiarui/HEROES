import glob


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
            for _ in range(seq_len):
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


if __name__ == '__main__':
    all_files = glob.glob("../Criteo/t*/*.txt")
    print(all_files)
    for single_file in all_files:
        out_file = single_file + 'small'
        with open(single_file) as f:
            with open(out_file, 'w') as out_f:
                for i in range(1000):
                    tmp = f.readline()
                    out_f.write(tmp)
                    (seq_len, label) = [int(_) for _ in tmp.rstrip().split()]
                    for j in range(seq_len):
                        line = f.readline()
                        out_f.write(line)
