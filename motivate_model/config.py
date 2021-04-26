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

configure_list['Criteo']['motivate-single'] = {
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
configure_list['alicpp']['motivate-single'] = {
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
    'l2_reg': 3e-2
}

configure_list['Criteo']['RRN'] = {
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
    'l2_reg': 1e-2,
    'position_embed': False
}
configure_list['alicpp']['RRN'] = {
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
    'l2_reg': 3e-2
}

configure_list['Criteo']['time_LSTM'] = {
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
    'l2_reg': 1e-2,
    'position_embed': False
}
configure_list['alicpp']['time_LSTM'] = {
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
    'l2_reg': 3e-2
}

configure_list['Criteo']['STAMP'] = {
    'embedding_size': 32,
    'seq_max_len': 50,
    'max_features': 5897,
    'n_hidden': 128,
    'n_classes': 1,
    'batch_size': 100,
    'keep_prob': 0.5,
    'prediction_embed_list': '64,32,16',
    'decay_step': 8000,
    'lr': 1e-3,
    'click_weight': 0.5,
    'conversion_weight': 0.1,
    'l2_reg': 1e-2,
    'time_stamp': False,
    'position_embed': False
}
configure_list['alicpp']['STAMP'] = {
    'embedding_size': 32,
    'seq_max_len': 160,
    'max_features': 638072,
    'n_hidden': 128,
    'n_classes': 1,
    'batch_size': 50,
    'keep_prob': 0.4,
    'prediction_embed_list': '64,32,16',
    'decay_step': 400,
    'lr': 1e-3,
    'click_weight': 0.14,
    'conversion_weight': 0.023,
    'l2_reg': 1e-2,
    'position_embed': False
}


def get_exp_configure(args):
    return configure_list[args['dataset']][args['model']]
