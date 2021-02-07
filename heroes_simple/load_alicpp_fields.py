import glob
import pickle

import numpy as np
from collections import defaultdict

Common_Fileds = {'101': '638072', '121': '638073', '122': '638074', '124': '638075', '125': '638076', '126': '638077',
                 '127': '638078', '128': '638079', '129': '638080', '205': '638081', '301': '638082'}
UMH_Fileds = {'109_14': ('u_cat', '638083'), '110_14': ('u_shop', '638084'), '127_14': ('u_brand', '638085'),
              '150_14': ('u_int', '638086')}  # user multi-hot feature
Ad_Fileds = {'206': ('a_cat', '638087'), '207': ('a_shop', '638088'), '210': ('a_int', '638089'),
             '216': ('a_brand', '638090')}  # ad feature for DIN
X_Fileds = {'508': ('x_a', '638091'), '509': ('x_b', '638092'), '702': ('x_c', '638093'), '853': ('x_d', '638094')}


def loadAliBatch(max_seq_len, fin, batch_seq_list):
    total_data_id = defaultdict(lambda: [])
    total_data_value = defaultdict(lambda: [])
    total_seqlen = []
    total_click = []
    total_label = []
    for seq_len in batch_seq_list:
        click = []
        label = []
        try:
            if seq_len > max_seq_len:
                for i in range(seq_len - max_seq_len):
                    fin.readline()
            for i in range(min(seq_len, max_seq_len)):
                line = fin.readline().rstrip().split()
                splits = [_.split(':') for _ in line[6:]]
                splits = np.reshape(splits, (-1, 3))
                tmp_id = np.array([])
                for f, def_id in Common_Fileds.items():
                    if f in splits[:, 0]:
                        mask = np.array(f == splits[:, 0])
                        tmp_id = np.append(tmp_id, splits[mask, 1])
                    else:
                        tmp_id = np.append(tmp_id, def_id)
                total_data_id['feat_ids'].append(tmp_id)
                for f, (fname, def_id) in UMH_Fileds.items():
                    if f in splits[:, 0]:
                        mask = np.array(f == splits[:, 0])
                        tmp_id = splits[mask, 1]
                        tmp_value = splits[mask, 2]
                    else:
                        tmp_id = np.array([def_id])
                        tmp_value = np.array([1.0])
                    total_data_id[fname + 'ids'].append(tmp_id.astype(np.int))
                    total_data_value[fname + 'vals'].append(tmp_value.astype(np.float))
                for f, (fname, def_id) in Ad_Fileds.items():

                    if f in splits[:, 0]:
                        mask = np.array(f == splits[:, 0])
                        tmp_id = splits[mask, 1]
                    else:
                        tmp_id = np.array([def_id])
                    total_data_id[fname + 'ids'].append(tmp_id.astype(np.int))
                for f, (fname, def_id) in X_Fileds.items():
                    if f in splits[:, 0]:
                        mask = np.array(f == splits[:, 0])
                        tmp_id = splits[mask, 1]
                        tmp_value = splits[mask, 2]
                    else:
                        tmp_id = np.array([def_id])
                        tmp_value = np.array([1.0])
                    total_data_id[fname + 'ids'].append(tmp_id.astype(np.int))
                    total_data_value[fname + 'vals'].append(tmp_value.astype(np.float))
                click.append([int(line[2])])
                label.append([int(line[3])])
            if seq_len < max_seq_len:
                for _ in range(seq_len, max_seq_len):
                    for k in total_data_id.keys():
                        if k == 'feat_ids':
                            total_data_id[k].append(np.array([0] * 11).astype(np.int))
                        else:
                            total_data_id[k].append(np.array([0]).astype(np.int))
                    for k in total_data_value.keys():
                        total_data_value[k].append(np.array([1.0]).astype(np.float))
                    click.append([0])
                    label.append([0])
            else:
                seq_len = max_seq_len
        except Exception as e:
            print(e)
            continue
        total_click.append(click)
        total_label.append(label)
        total_seqlen.append(seq_len)
    return dict(total_data_id), dict(total_data_value), total_click, total_label, total_seqlen


if __name__ == '__main__':
    '''h = ["1.1:1.1:1.1", "2.1:2.1:2.1"]
    h1 = [_.split(':') for _ in h]
    print(h1)
    tmp_id = []
    tmp_id.append(np.reshape(h1, (-1, 3))[:, 1].astype(np.float))
    print("Why", tmp_id)
    h2 = np.reshape(h1, (-1, 3)).astype(np.float)
    print(h2.tolist())
    print(h2, type(h2))
    print(h2[:, 2])
    print(h2[:, 2][-1:])
    j = [h2, [[3, 3, 3], [4, 4, 4]], [[5, 6, 7], [8, 9, 10]]]
    print(j)
    print(np.reshape(j, (-1, 3)))
    import pickle

    with open('../alicpp/train/remap_sample/r_train.txt' + '.pkl', 'rb') as len_f:
        data = list(pickle.load(len_f))
    print(max(data))
    for i in [1, 10, 20, 40, 80, 100, 160, 200]:
        print(np.sum(list(map(lambda x: x >= i, data))))


    # 转化一个序列列表为稀疏矩阵
    def sparse_tuple_from(sequences, dtype=np.int32):
        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), indices[:, 1].max() + 1], dtype=np.int64)
        return indices, values, shape


    indices, values, shape = sparse_tuple_from(
        [[7, 4, 11, 11, 14], [19, 14, 12], [19, 14, 12, 10, 4, 11, 11, 14], [19, 14, 12], [0], [0], [0], [0]])
    v_indices, v_values, v_shape = sparse_tuple_from(
        [[7, 4, 11, 11, 14], [19, 14, 12], [19, 14, 12, 10, 4, 11, 11, 14], [19, 14, 12], [1], [1], [1], [0.1]],
        np.float32)
    sparse_tuple_from([])
    print(indices, values, shape)
    import tensorflow as tf

    input_id = tf.sparse_placeholder(tf.int32, shape=[None, None], name='id')
    input_value = tf.sparse_placeholder(tf.float32, shape=[None, None], name='id')
    embedding_matrix = tf.Variable(
        tf.random_normal([25, 10], stddev=0.1))
    out = tf.nn.embedding_lookup_sparse(embedding_matrix, sp_ids=input_id, sp_weights=input_value)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        h = sess.run(out, feed_dict={input_id: (indices, values, shape), input_value: (v_indices, v_values, v_shape)})
        print(h)
        print(h.shape)
        print(h.reshape(-1, 2, h.shape[1]))'''
    all_files = glob.glob("./../alicpp/small/small*.txt")
    print("all_files:", all_files)
    for tr in all_files:
        tr_len = tr + '.pkl'
        with open(tr_len, 'rb') as len_f:
            tr_len_list = list(pickle.load(len_f))
        print(tr_len_list)
        print(len(tr_len_list))
