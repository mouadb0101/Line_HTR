from dataset import readIAMData
from config import *
from model import get_model
from utils import str2list, map_and_count
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import time


@tf.function
def train_one_step(model, X, Y, label_length, optimizer):
    trainable_vars = model.trainable_variables
    with tf.GradientTape() as tape:
        inp = {
            'the_input': X,
            # 'A_in':fltr
        }
        y_pred = model(X, training=True)
        # print(y_pred)

        logit_length = np.array([y_pred.shape[1]] * y_pred.shape[0]).reshape((y_pred.shape[0], 1))
        loss = keras.backend.ctc_batch_cost(Y, y_pred, logit_length, label_length)
        loss = tf.reduce_mean(loss)
    # Compute first-order gradients
    grads = tape.gradient(loss, trainable_vars)
    # print(grads)

    optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_vars))
    return loss


@tf.function
def val_one_step(model, X, Y, label_length):
    inp = {
        'the_input': X,
    }

    y_pred = model(X, training=False)
    # print(y_pred.get_shape())

    logit_length = np.array([y_pred.shape[1]] * y_pred.shape[0]).reshape((y_pred.shape[0], 1))
    loss = keras.backend.ctc_batch_cost(Y, y_pred, logit_length, label_length)  # ["softmax"]
    loss = tf.reduce_mean(loss)

    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                       sequence_length=[y_pred.shape[1]] * y_pred.shape[0],
                                                       merge_repeated=True)

    """decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                               sequence_length=[y_pred.shape[1]]*y_pred.shape[0],
                                               beam_width=8,
                                               top_paths=1)"""

    return decoded, loss


if __name__ == "__main__":
    data = readIAMData(train_path, (img_w, img_h))
    print(len(data))

    dataTest = readIAMData(test_path, (img_w, img_h))
    print(len(dataTest))

    x_train = list(data.values())
    y_train = [str2list(s, maxTextLen) for s in data.keys()]

    x_test = list(dataTest.values())
    y_test = [str2list(s, maxTextLen) for s in dataTest.keys()]

    x_train = tf.keras.backend.expand_dims(x_train, axis=-1)
    y_train = np.array(y_train)

    x_test = tf.keras.backend.expand_dims(x_test, axis=-1)
    y_test = np.array(y_test)

    model = get_model()

    # Instantiate a metric object
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # optimizer
    optimizer = tfa.optimizers.AdamW(weight_decay=wd, learning_rate=lr)

    tf.keras.backend.clear_session()

    avg_loss = tf.keras.metrics.Mean(name="train_loss")
    val_avg_loss = tf.keras.metrics.Mean(name="val_loss")
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    last_checkpoint_path = ''

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if last_checkpoint_path != '':
        localtime = last_checkpoint_path.rstrip("/").split("/")[-1]
    manager = tf.train.CheckpointManager(checkpoint, directory="output/tf_ckpts_{}".format(localtime),
                                         max_to_keep=max_to_keep)
    summary_writer = tf.summary.create_file_writer("output/{}".format(localtime))
    checkpoint.restore(manager.latest_checkpoint)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

    else:
        print("Initializing from scratch")

    for epoch in range(epochs):

        # Iterate over the batches of a dataset.
        if epoch % 5:
            mx = mx * 2
        for step, (x, y) in enumerate(train_dataset):
            # print(y[0])        ["softmax"]
            label_length = np.array([min(len(i[i != num_classes]), mx) for i in y]).reshape((len(y), 1))

            loss = train_one_step(model, x, y, label_length, optimizer)
            # print('batch [{}]: {:.2f}'.format(step + 1, loss))

            # tf.summary.scalar("train_loss", loss, step=optimizer.iterations)
            avg_loss.update_state(loss)
        print("[{} / {}] Mean train loss: {}".format(epoch + 1, epochs, avg_loss.result()))
        avg_loss.reset_states()
        if epoch % 1 == 0:
            if 1:
                num_correct_samples = 0
                wer = 0
                cer = 0
                dataset_len = 0
                for step, (X, Y) in enumerate(test_dataset):
                    label_length = np.array([min(len(i[i != num_classes]), mx) for i in Y]).reshape((len(Y), 1))
                    decoded, loss = val_one_step(model, X, Y, label_length, False)  # ["softmax"]
                    count_s, count_w, count_c = map_and_count(decoded, Y, wordCharList, num_classes, show=False)
                    val_avg_loss.update_state(loss)
                    num_correct_samples += count_s
                    cer += count_c
                    wer += count_w
                    dataset_len += 1 * Y.get_shape()[0]
                # tf.summary.scalar("val_loss", val_avg_loss.result(), step=epoch)

                tf.summary.scalar("accuracy(line, greedy decoder)", num_correct_samples / dataset_len, step=epoch)
                print("[{} / {}] Mean val loss: {}".format(epoch + 1, epochs, val_avg_loss.result()))
                print("[{} / {}] Accuracy(line, greedy decoder): {:.2f}, WER: {:.2f}, CER: {:.2f}".format(epoch + 1,
                                                                                                          epochs,
                                                                                                          num_correct_samples / dataset_len,
                                                                                                          wer * 100 / dataset_len,
                                                                                                          cer * 100 / dataset_len))
                saved_path = manager.save(checkpoint_number=epoch)
                print("Model saved to {}".format(saved_path))

                if val_avg_loss.result() < best_ctc:
                    print('Best ctc: {:.2f}'.format(val_avg_loss.result()))
                    model.save_weights(
                        "{weight_path}/best_ctc_{model_name}".format(weight_path=weight_path, model_name=model_name))
                    best_ctc = val_avg_loss.result()

                if (cer / dataset_len) < best_cer:
                    print('Best cer: {:.2f}'.format(cer * 100 / dataset_len))
                    model.save_weights(
                        "{weight_path}/best_cer_{model_name}".format(weight_path=weight_path, model_name=model_name))
                    best_cer = (cer / dataset_len)

                print('')
                print('')
                val_avg_loss.reset_states()

    model.save_weights("{weight_path}/last_{model_name}".format(weight_path=weight_path, model_name=model_name))

    num_correct_samples = 0
    wer = 0
    cer = 0
    dataset_len = 0
    model.load_weights("{weight_path}/best_cer_{model_name}".format(weight_path=weight_path, model_name=model_name))
    for step, (X, Y) in enumerate(test_dataset):
        label_length = np.array([len(i[i != num_classes]) for i in Y]).reshape((len(Y), 1))
        decoded, loss = val_one_step(model, X, Y, label_length, attention=False)
        count_s, count_w, count_c = map_and_count(decoded, Y, wordCharList, num_classes, show=True)
        num_correct_samples += count_s
        cer += count_c
        wer += count_w
        dataset_len += 1 * Y.get_shape()[0]
    print("[{} / {}] Accuracy(line, greedy decoder): {:.2f}, WER: {:.2f}, CER: {:.2f}".format(epoch + 1, epochs,
                                                                                              num_correct_samples / dataset_len,
                                                                                              wer * 100 / dataset_len,
                                                                                              cer * 100 / dataset_len))
