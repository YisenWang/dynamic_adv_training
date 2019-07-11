# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os, time
import numpy as np
import keras.backend as K
import tensorflow as tf
from datasets import get_data
from models import get_model
from losses import cross_entropy
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from pgd_attack import LinfPGDAttack, TestLinfPGDAttack
from logger import Logger
from tqdm import tqdm
import time

# prepare folders
folders = ['data', 'model', 'log']
for folder in folders:
    path = os.path.join('./', folder)
    if not os.path.exists(path):
        os.makedirs(path)

def advs_train(dataset='cifar-10', loss_name='ce', epochs=120, dynamic_epoch=100,
               batch_size=128, fosc_max=0.5, epsilon=0.031):
    """
    Adversarial training with PGD attack.
    """
    print('DynamicAdvsTrain - Data set: %s, loss: %s, epochs: %s, dynamic_epoch: %s, batch: %s, epsilon: %s' %
          (dataset, loss_name, epochs, dynamic_epoch, batch_size, epsilon))

    X_train, Y_train, X_test, Y_test = get_data(dataset, clip_min=0., clip_max=1., onehot=True)

    n_images = X_train.shape[0]
    image_shape = X_train.shape[1:]
    n_class = Y_train.shape[1]
    print("n_images:", n_images, "n_class:", n_class, "image_shape:", image_shape)

    model = get_model(dataset, input_shape=image_shape, n_class=n_class, softmax=True)
    # model.summary()

    # create loss
    if loss_name == 'ce':
        loss = cross_entropy
    else:
        print("New loss function should be defined first.")
        return

    optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # data augmentation
    if dataset in ['mnist']:
        datagen = ImageDataGenerator()
    elif dataset in ['cifar-10']:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    datagen.fit(X_train)

    # pgd attack for training
    attack = LinfPGDAttack(model,
                           epsilon=epsilon,
                           eps_iter=epsilon/4,
                           nb_iter=10,
                           random_start=True,
                           loss_func='xent',
                           clip_min=np.min(X_train),
                           clip_max=np.max(X_train))

    # initialize logger
    mylogger = Logger(K.get_session(), model, X_train, Y_train, X_test, Y_test,
                      dataset, loss_name, epochs, suffix='%s' % epsilon)

    batch_iterator = datagen.flow(X_train, Y_train, batch_size=batch_size)
    
    start_time = time.time()

    for ep in range(epochs):        
        # learning rate decay
        if (ep + 1) == 60:
            lr = float(K.get_value(model.optimizer.lr))
            K.set_value(model.optimizer.lr, lr/10.0)
            
        if (ep + 1) == 100:
            lr = float(K.get_value(model.optimizer.lr))
            K.set_value(model.optimizer.lr, lr/10.0)
        lr = float(K.get_value(model.optimizer.lr))

        # a simple linear decreasing of fosc
        fosc = fosc_max - fosc_max * (ep*1.0/dynamic_epoch)
        fosc = np.max([fosc, 0.0])

        steps_per_epoch = int(X_train.shape[0]/batch_size)
        pbar = tqdm(range(steps_per_epoch))
        for it in pbar:
            batch_x, batch_y = batch_iterator.next()
            batch_advs, fosc_batch = attack.perturb(K.get_session(), batch_x, batch_y, batch_size, ep, fosc)
            
            probs = model.predict(batch_advs)
            loss_weight = np.max(- batch_y * np.log(probs + 1e-12), axis = 1)
            
            if it == 0:
                fosc_all = fosc_batch
            else:
                fosc_all = np.concatenate((fosc_all, fosc_batch), axis=0)
                
            if ep == 0:
                loss, acc = model.train_on_batch(batch_advs, batch_y)
            else:
                loss, acc = model.train_on_batch(batch_advs, batch_y, sample_weight = loss_weight)
            pbar.set_postfix(acc='%.4f' % acc, loss='%.4f' % loss)
            
        print('All time:', time.time() - start_time)
        
        log_path = './log'

        file_name = os.path.join(log_path, 'BatchSize_{}_Epoch_{}_fosc.npy'.format(batch_size, ep))
        np.save(file_name, fosc_all)

        val_loss, val_acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
        logs = {'acc': acc, 'loss': loss, 'val_acc': val_acc, 'val_loss': val_loss}

        print("Epoch %s - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f"
              % (ep, loss, acc, val_loss, val_acc))

        # save the log and model every epoch
        mylogger.on_epoch_end(epoch=ep, logs=logs)
        model.save_weights("model/advs_%s_%s_%s_%s.hdf5" % (dataset, loss_name, epsilon, ep))
        

def main(args):
    """
    Train model with data augmentation: random padding+cropping and horizontal flip
    :param args:
    :return:
    """
    advs_train(args.dataset, args.loss, args.epochs, args.dynamic_epoch,
               args.batch_size, args.fosc_max, args.epsilon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar-10'",
        required=True, type=str
    )
    parser.add_argument(
        '-l', '--loss',
        help="loss name: 'ce'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-t', '--dynamic_epoch',
        help="The maximum control epoch for dynamic advs training.",
        required=False, type=float
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-p', '--epsilon',
        help="The maximum perturbation.",
        required=False, type=float
    )
    parser.add_argument(
        '-fm', '--fosc_max',
        help="The maximum perturbation.",
        required=False, type=float
    )
    parser.set_defaults(epochs=120)
    parser.set_defaults(dynamic_epoch=100)
    parser.set_defaults(batch_size=128)
    parser.set_defaults(fosc_max=0.5)

    # pass in arguments from the command line
    # args = parser.parse_args()
    # main(args)

    # set parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use the fisrt GPU.
    args = parser.parse_args(['-d', 'cifar-10', '-l', 'ce', '-e', '120', '-t', '100',
                              '-b', '128', '-fm', '0.5', '-p', '0.031'])
    main(args)
    K.clear_session()
