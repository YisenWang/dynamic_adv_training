# -*- coding: utf-8 -*-

import os
import numpy as np
import keras.backend as K
from pgd_attack import TestLinfPGDAttack
import tensorflow as tf

# maximum perturbation constraints for testing
EPSILONS = {'mnist': 0.3, 'cifar-10': 0.031}

class Logger():
    """
    Log train/val loss and acc into file for later plots.
    """
    def __init__(self, sess, model, X_train, y_train, X_test, Y_test,
                 dataset, loss_name, epochs, suffix):
        self.sess = sess
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.dataset = dataset
        self.loss_name = loss_name
        self.epochs = epochs
        self.log_path = 'log'
        self.suffix = suffix # give the file a customized name

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        # robustness metrics: 2) FGSM Score 3) PGD Score
        self.batch_size = 100

        self.fgsms = []  # FGSM - 1 step PGD
        self.fgsm = TestLinfPGDAttack(model,
                                  epsilon=EPSILONS[self.dataset],
                                  eps_iter=EPSILONS[self.dataset],
                                  nb_iter=1,
                                  random_start=True,
                                  loss_func='xent',
                                  clip_min=np.min(self.X_train),
                                  clip_max=np.max(self.X_train))

        self.pgds = []  # PGD Score
        self.pgd = TestLinfPGDAttack(model,
                                 epsilon=EPSILONS[self.dataset],
                                 eps_iter=EPSILONS[self.dataset]/4.,
                                 nb_iter=20,
                                 random_start=True,
                                 loss_func='xent',
                                 clip_min=np.min(self.X_train),
                                 clip_max=np.max(self.X_train))
        
    def on_epoch_end(self, epoch, logs={}):
        tr_acc = logs.get('acc')
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')

        self.train_loss.append(tr_loss)
        self.test_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(val_acc)

        # you can select a subset from testset to save sometimes, but may result in high variance.
        X_set = self.X_test
        Y_set = self.Y_test

        ## compute FGSM score
        X_adv = self.fgsm.perturb(self.sess, X_set, Y_set, self.batch_size)
        # statistics of the attacks
        _, acc = self.model.evaluate(X_adv, Y_set, batch_size=self.batch_size, verbose=0)
        self.fgsms.append(acc)

        # jsut print the latest five values
        if len(self.fgsms) > 5:
            print('FGSM = ..., ', np.array(self.fgsms)[-5:])
        else:
            print('FGSM = ', np.array(self.fgsms))


        ## compute PGD score
        X_adv = self.pgd.perturb(self.sess, X_set, Y_set, self.batch_size)
        # statistics of the attacks
        _, acc = self.model.evaluate(X_adv, Y_set, batch_size=self.batch_size, verbose=0)
        self.pgds.append(acc)

        # jsut print the latest five values
        if len(self.pgds) > 5:
            print('PGD = ..., ', np.array(self.pgds)[-5:])
        else:
            print('PGD = ', np.array(self.pgds))

        file_name = os.path.join(self.log_path, 'train_stats_%s_%s_%s_%s.npy' % (self.dataset, self.loss_name, self.suffix, epoch))
        np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss),
                                      np.array(self.train_acc), np.array(self.test_acc),
                                      np.array(self.fgsms), np.array(self.pgds))))

        return
