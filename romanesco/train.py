#!/usr/bin/env python3

import os
import logging

import numpy as np
import tensorflow as tf

from romanesco import reader
from romanesco.const import *
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph
####
from romanesco.score import score
import sys

def train(data: str, epochs: int, batch_size: int, vocab_max_size: int,
          save_to: str, log_to: str, **kwargs):
    """Trains a language model. See argument description in `bin/romanesco`."""

    # create folders for model and logs if they don't exist yet
    for folder in [save_to, log_to]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # create vocabulary to map words to ids
    vocab = Vocabulary()
    vocab.build(data, max_size=vocab_max_size)
    vocab.save(os.path.join(save_to, VOCAB_FILENAME))

    # convert training data to list of word ids
    raw_data = reader.read(data, vocab)

    # define computation graph
    inputs, targets, loss, train_step, _, summary = define_computation_graph(vocab.size, batch_size)

    saver = tf.train.Saver()

    dev_data = reader.read(kwargs.pop("dev"), vocab)

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())


        # iterate over training data `epoch` times
        for epoch in range(1, epochs + 1):
            ###
            # iteration_perplexities = []
            ###

            total_loss = 0.0
            total_iter = 0
            for x, y in reader.iterate(raw_data, batch_size, NUM_STEPS):
                l, _, s = session.run([loss, train_step, summary],
                                      feed_dict={inputs: x, targets: y})
                summary_writer.add_summary(s, total_iter)
                total_loss += l
                total_iter += 1
                if total_iter % 100 == 0:
                    logging.debug("Epoch=%s, iteration=%s", epoch, total_iter)

                    ### Attempt to add early stop functionality.
                    # For each 100th iteration, score perplexity on dev set. If the current iteration perplexity is greater than the last 5 perplexity scores in list 'iteration perplexities', model is starting to overfit.
                    # current_iter_per = score(dev, load_from, batch_size)

                    # iteration_perplexities.append(current_iter_per)
                    # if len(iteration_perplexity) >= 5:
                    #     for i in iteration_perplexity[-5:]:
                    #         if current_iter_per > i:
                    #             perplexity = np.exp(total_loss / total_iter)
                    #             logging.info("Lowest perplexity reached. Training has been early stopped. Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
                    #             saver.save(session, os.path.join(save_to, MODEL_FILENAME))
                    #             sys.exit()

                    total_loss_dev = 0.0
                    total_iter_dev = 0

                    # Set previous iter per at a high perplexity
                    previous_iter_per = float(1000)
                    # Set counter to record number of times
                    c = 0

                    for x, y in reader.iterate(dev_data,
                                               batch_size,
                                               NUM_STEPS):

                        l = session.run([loss], feed_dict={inputs: x,
                                                           targets: y})

                        total_loss_dev += l[0]
                        total_iter_dev += 1

                    current_iter_per = np.exp(total_loss_dev / total_iter_dev)

                    logging.info("current perplexity on dev set: %.2f",
                                 current_iter_per)

                    if current_iter_per > previous_iter_per:
                        c += 1
                    else:
                        c = 0

                    if c > 3:

                    # Update previous iteration perpexity with current and reset current
                        perplexity = np.exp(total_loss / total_iter)
                        logging.info("Training stopped early. Model starting to overfit. Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
                        saver.save(session, os.path.join(save_to, MODEL_FILENAME))
                        sys.exit()

                    previous_iter_per = current_iter_per
                    del current_iter_per

                    # iteration_perplexities.append(current_iter_per)
                    # for i in iteration_perplexities[-5:]:
                    #     if current_iter_per > i:
                    #         perplexity = np.exp(total_loss / total_iter)
                    ###

            perplexity = np.exp(total_loss / total_iter)
            logging.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
            saver.save(session, os.path.join(save_to, MODEL_FILENAME))
