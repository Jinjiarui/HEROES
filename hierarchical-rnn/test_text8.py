from hmlstm import HMLSTMNetwork, prepare_inputs, get_text
from hmlstm.viz import viz_char_boundaries

batches_in, batches_out = prepare_inputs(batch_size=10, truncate_len=5000,
                                         step_size=2500, text_path='text8.txt')

network = HMLSTMNetwork(output_size=27, input_size=27, embed_size=2048,
                        out_hidden_size=1024, hidden_state_sizes=1024,
                        task='classification')

network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True,
              load_vars_from_disk=False, variable_path='./text8')

predictions = network.predict(batches_in[-1], variable_path='./text8')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./text8')

# visualize boundaries
viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])
