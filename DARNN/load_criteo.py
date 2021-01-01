def loadCriteoBatch(batchsize, max_seq_len, max_input, fin, min_input=3):
    total_data = []
    total_seqlen = []
    total_label = []
    i = 0
    while i < batchsize:
        tmp_seq = []
        try:
            (seq_len, label) = [int(_) for _ in fin.readline().rstrip().split()]
            for _ in range(seq_len):
                line = fin.readline().rstrip().split()
                tmp_seq.append([float(line[0])] + [int(_) for _ in line[1:]])
            if seq_len < min_input:
                continue
            i += 1
            # 大于最大长度的截断，取后面存在conversion的部分
            tmp_seq = tmp_seq[-max_seq_len:]
            # 小于最大长度的补全
            if seq_len < max_seq_len:
                for _ in range(seq_len, max_seq_len):
                    tmp_seq.append([0] * max_input)
            else:
                seq_len = max_seq_len
        except:
            i += 1
            continue
        total_data.append(tmp_seq)
        total_seqlen.append(seq_len)
        total_label.append(label)
    return total_data, total_seqlen, total_label


def loaddualattention(batchsize, max_seq_len, max_input, fin, min_input=3):
    total_data, total_seqlen, total_label = loadCriteoBatch(batchsize, max_seq_len, max_input, fin, min_input)
    batchsize = len(total_data)
    click_label = []
    for i in range(batchsize):
        click = []
        for j in range(total_seqlen[i]):
            if total_data[i][j][1] == 1:
                click.append([0, 1])
            else:
                click.append([1, 0])
        click_label.append(click + [[0, 0]] * (max_seq_len - total_seqlen[i]))
    return total_data, click_label, total_label, total_seqlen
