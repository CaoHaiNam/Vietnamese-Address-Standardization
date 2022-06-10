# from Parameters import *
# import tensorflow as tf
# import keras
# import tensorflow_hub as hub
# import bert
# import numpy as np

# def get_model(model_url, max_seq_length):
#   labse_layer = hub.KerasLayer(model_url, trainable=True)

#   # Define input.
#   input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                          name="input_word_ids")
#   input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                      name="input_mask")
#   segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                       name="segment_ids")

#   # LaBSE layer.
#   pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

#   # The embedding is l2 normalized.
#   pooled_output = tf.keras.layers.Lambda(
#       lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

#   # Define model.
#   return tf.keras.Model(
#         inputs=[input_word_ids, input_mask, segment_ids],
#         outputs=pooled_output), labse_layer

# # labse_model, labse_layer = get_model(model_url, max_seq_length=max_seq_length)

# def create_input(input_strings, tokenizer, max_seq_length):

#   input_ids_all, input_mask_all, segment_ids_all = [], [], []
#   for input_string in input_strings:
#     # Tokenize input.
#     input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
#     input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
#     sequence_length = min(len(input_ids), max_seq_length)

#     # Padding or truncation.
#     if len(input_ids) >= max_seq_length:
#       input_ids = input_ids[:max_seq_length]
#     else:
#       input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

#     input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

#     input_ids_all.append(input_ids)
#     input_mask_all.append(input_mask)
#     segment_ids_all.append([0] * max_seq_length)

#   return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)
