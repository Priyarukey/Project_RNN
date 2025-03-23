import os
import numpy as np
import re
import shutil
import tensorflow as tf

DATA_DIR = "./data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")


def clean_logs():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    shutil.rmtree(LOG_DIR, ignore_errors=True)


def download_and_read(urls):
    texts = []
    for i, url in enumerate(urls):
        p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url,
            cache_dir=".")
        text = open(p, mode="r", encoding="utf-8").read()
        # remove byte order mark
        text = text.replace("\ufeff", "")
        # remove newlines
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', " ", text)
        # add it to the list
        texts.extend(text)
    return texts


def split_train_labels(sequence):
    input_seq = sequence[0:-1]
    output_seq = sequence[1:]
    return input_seq, output_seq


class CharGenModel(tf.keras.Model):
    def __init__(self, vocab_size, sequence_length, embedding_dim, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        # Convert integer-encoded characters into dense embeddings.
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        # GRU layer that retains state between batches for sequence continuity.
        self.gru = tf.keras.layers.GRU(
    units=sequence_length,
    recurrent_initializer="glorot_uniform",  # Initializes the recurrent weights using Glorot uniform.
    recurrent_activation="sigmoid",            # Applies the sigmoid function to the recurrent connections.
    stateful=True,
    return_sequences=True
)
        # Fully connected layer to project GRU outputs to vocabulary size.
        self.dense = tf.keras.layers.Dense(units=vocab_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.gru(x)
        return self.dense(x)


    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer(x)
        return x


def loss_fn(labels, predictions):
    # Compute the sparse categorical crossentropy loss.
    # 'from_logits=True' indicates that 'predictions' are raw scores (logits)
    # and a softmax will be applied internally.
    return tf.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=True)



def generate_text(model, start_string, char_to_index, index_to_char,
                  gen_length=1000, temperature=1.0):
    # Convert the starting string to a list of indices
    input_indices = [char_to_index[char] for char in start_string]
    # Create a batch of one sequence
    input_tensor = tf.expand_dims(input_indices, 0)

    generated_text = []
    # Reset the model's state before generation
    model.reset_states()

    for _ in range(gen_length):
        # Get the model's predictions
        predictions = model(input_tensor)
        # Remove the batch dimension and adjust by the temperature
        predictions = tf.squeeze(predictions, axis=0) / temperature
        # Sample an index from the predictions
        sampled_index = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # Append the corresponding character to the generated text list
        generated_text.append(index_to_char[sampled_index])
        # Use the sampled index as the next input to the model
        input_tensor = tf.expand_dims([sampled_index], 0)

    # Combine the start string with the generated characters
    return start_string + "".join(generated_text)


# download and read into local data structure (list of chars)
texts = download_and_read([
    "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
    "https://www.gutenberg.org/files/12/12-0.txt"
])
clean_logs()

# create the vocabulary
vocab = sorted(set(texts))
print("vocab size: {:d}".format(len(vocab)))

# create mapping from vocab chars to ints
char2idx = {c:i for i, c in enumerate(vocab)}
idx2char = {i:c for c, i in char2idx.items()}

# numericize the texts
texts_as_ints = np.array([char2idx[c] for c in texts])
data = tf.data.Dataset.from_tensor_slices(texts_as_ints)

# number of characters to show before asking for prediction
# sequences: [None, 100]
seq_length = 100
sequences = data.batch(seq_length + 1, drop_remainder=True)
sequences = sequences.map(split_train_labels)

# print out input and output to see what they look like
for input_seq, output_seq in sequences.take(1):
    print("input:[{:s}]".format(
        "".join([idx2char[i] for i in input_seq.numpy()])))
    print("output:[{:s}]".format(
        "".join([idx2char[i] for i in output_seq.numpy()])))

# set up for training
# batches: [None, 64, 100]
batch_size = 64
steps_per_epoch = len(texts) // seq_length // batch_size
dataset = sequences.shuffle(10000).batch(batch_size, drop_remainder=True)
print(dataset)

# define network
vocab_size = len(vocab)
embedding_dim = 256

model = CharGenModel(vocab_size, seq_length, embedding_dim)
model.build(input_shape=(batch_size, seq_length))
model.summary()

# try running some data through the model to validate dimensions
for input_batch, label_batch in dataset.take(1):
    pred_batch = model(input_batch)

print(pred_batch.shape)
assert(pred_batch.shape[0] == batch_size)
assert(pred_batch.shape[1] == seq_length)
assert(pred_batch.shape[2] == vocab_size)

model.compile(optimizer=tf.optimizers.Adam(), loss=loss)

# we will train our model for 50 epochs, and after every 10 epochs
# we want to see how well it will generate text
num_epochs = 50
for i in range(num_epochs // 10):
    model.fit(
        dataset.repeat(),
        epochs=10,
        steps_per_epoch=steps_per_epoch
        # callbacks=[checkpoint_callback, tensorboard_callback]
    )
    checkpoint_file = os.path.join(
        CHECKPOINT_DIR, "model_epoch_{:d}".format(i+1))
    model.save_weights(checkpoint_file)

    # create a generative model using the trained model so far
    gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
    gen_model.load_weights(checkpoint_file)
    gen_model.build(input_shape=(1, seq_length))

    print("after epoch: {:d}".format(i+1)*10)
    print(generate_text(gen_model, "Alice ", char2idx, idx2char))
    print("---")
