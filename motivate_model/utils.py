import numpy as np
import full_model


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


def load_model(placeholders, args):
    time_stamp = 'time_stamp' not in args.keys()
    position_embed = 'position_embed' not in args.keys()
    model = full_model.Model(placeholders,
                             embedding_size=args['embedding_size'],
                             seq_max_len=args['seq_max_len'],
                             max_features=args['max_features'],
                             n_hidden=args['n_hidden'],
                             n_classes=args['n_classes'],
                             keep_prob=args['keep_prob'],
                             prediction_embed_list=[int(i) for i in args['prediction_embed_list'].split(',')],
                             decay_step=args['decay_step'],
                             lr=args['lr'],
                             click_weight=args['click_weight'],
                             conversion_weight=args['conversion_weight'],
                             ctr_task_wgt=args['ctr_task_wgt'],
                             l2_reg=args['l2_reg'],
                             position_embed=position_embed,
                             time_stamp=time_stamp,
                             model_name=args['model'],
                             dataset_name=args['dataset'])

    return model
