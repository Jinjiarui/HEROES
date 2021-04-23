configure_list = {'Criteo': {}, 'alicpp': {}}

configure_list['Criteo']['Heroes'] = {
    'embedding_size': 48,
    'seq_max_len': 50,
    'max_features': 5897,
    'n_hidden': 128,
    'n_classes': 1,
    'batch_size': 100,
    'keep_prob': 0.5,
    'prediction_embed_list': '64,32,16',
    'decay_step': 8000,
    'lr': 1e-2,
    'click_weight': 0.5,
    'conversion_weight': 0.1,
    'l2_reg': 1e-2
}
configure_list['alicpp']['Heroes'] = {
    'embedding_size': 32,
    'seq_max_len': 160,
    'max_features': 638072,
    'n_hidden': 128,
    'n_classes': 1,
    'batch_size': 50,
    'keep_prob': 0.5,
    'prediction_embed_list': '64,32,16',
    'decay_step': 400,
    'lr': 1e-2,
    'click_weight': 0.14,
    'conversion_weight': 0.023,
    'l2_reg': 1e-2
}

configure_list['Criteo']['motivate'] = {
    'embedding_size': 48,
    'seq_max_len': 50,
    'max_features': 5897,
    'n_hidden': 128,
    'n_classes': 1,
    'batch_size': 100,
    'keep_prob': 0.5,
    'prediction_embed_list': '64,32,16',
    'decay_step': 8000,
    'lr': 1e-2,
    'click_weight': 0.5,
    'conversion_weight': 0.1,
    'l2_reg': 1e-2
}
configure_list['alicpp']['motivate'] = {
    'embedding_size': 32,
    'seq_max_len': 160,
    'max_features': 638072,
    'n_hidden': 128,
    'n_classes': 1,
    'batch_size': 50,
    'keep_prob': 0.4,
    'prediction_embed_list': '64,32,16',
    'decay_step': 400,
    'lr': 5e-2,
    'click_weight': 0.14,
    'conversion_weight': 0.023,
    'l2_reg': 1e-1
}


def get_exp_configure(args):
    return configure_list[args['dataset']][args['model']]
