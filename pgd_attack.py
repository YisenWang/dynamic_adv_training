from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy import linalg as LA

class LinfPGDAttack:
    def __init__(self, model, epsilon, eps_iter, nb_iter, kappa=0, random_start=False,
                 loss_func='xent', clip_min=0.0, clip_max=1.0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.kappa = kappa
        self.rand = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.x_input = self.model.layers[0].input
        logits = self.model.layers[-2].output
        y_pred = tf.nn.softmax(logits)
        self.y_true = tf.placeholder(tf.float32, shape=y_pred.get_shape().as_list())

        if loss_func == 'xent':
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred+1e-20), axis=1)
        elif loss_func == 'cw':
            correct_logit = tf.reduce_sum(self.y_true * logits, axis=1)
            wrong_logit = tf.reduce_max((1 - self.y_true) * logits, axis=1)
            self.loss = -tf.nn.relu(correct_logit - wrong_logit + kappa)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred), axis=1)

        self.grad = tf.gradients(self.loss, self.x_input)[0]

    def perturb(self, sess, x_nat, y, batch_size, ep, cri):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        nb_batch = len(x) // batch_size
        # check if need one more batch
        if nb_batch * batch_size < len(x):
            nb_batch += 1

        fosc_batch = np.array([])
        for i in range(nb_batch):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = np.minimum(end, len(x))
            batch_x = x[start:end]
            batch_y = y[start:end]
            
            original_batch_x = np.copy(batch_x)
            step_size = self.eps_iter
            for j in range(self.nb_iter):
                if j == 0:
                    loss, grad = sess.run([self.loss, self.grad],
                                      feed_dict={self.x_input: batch_x,
                                                 self.y_true: batch_y})
                    batch_x += step_size * np.sign(grad)
                else:
                    batch_x += np.multiply(step_size * np.sign(grad),
                                           np.repeat(control_indicator, 32*32*3).reshape(batch_x.shape))

                batch_x = np.clip(batch_x, x_nat[start:end] - self.epsilon, x_nat[start:end] + self.epsilon)
                batch_x = np.clip(batch_x, self.clip_min, self.clip_max)  # ensure valid pixel range

                loss, grad = sess.run([self.loss, self.grad],
                                      feed_dict={self.x_input: batch_x,
                                                 self.y_true: batch_y})

                ## compute the FOSC criterion, the grad can be reused by next step perturbation.
                grad_adv = np.copy(grad)
                grad_flatten = grad_adv.reshape(batch_x.shape[0], -1)
                grad_norm = LA.norm(grad_flatten, ord=1, axis=1).reshape(-1, 1)
                diff = (batch_x - original_batch_x).reshape(batch_x.shape[0], -1)
                fosc = np.copy(grad_norm)
                for i in range(batch_x.shape[0]):
                    fosc[i] = - np.dot(grad_flatten[i], diff[i]) + self.epsilon * grad_norm[i]
                    
                control_indicator = np.copy(fosc)
                control_indicator[control_indicator<=cri] = 0
                control_indicator[control_indicator>cri] = 1
                    
                if j == 0:
                    fosc_batch = fosc
                else:
                    fosc_batch = np.concatenate((fosc_batch, fosc), axis=1)
                                   
            x[start:end] = batch_x[:]

        return x, fosc_batch


class TestLinfPGDAttack:
    def __init__(self, model, epsilon, eps_iter, nb_iter, kappa=0, random_start=False,
                 loss_func='xent', clip_min=0.0, clip_max=1.0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.kappa = kappa
        self.rand = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.x_input = self.model.layers[0].input
        logits = self.model.layers[-2].output
        y_pred = tf.nn.softmax(logits)
        self.y_true = tf.placeholder(tf.float32, shape=y_pred.get_shape().as_list())

        if loss_func == 'xent':
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred), axis=1)
        elif loss_func == 'cw':
            correct_logit = tf.reduce_sum(self.y_true * logits, axis=1)
            wrong_logit = tf.reduce_max((1 - self.y_true) * logits, axis=1)
            self.loss = -tf.nn.relu(correct_logit - wrong_logit + kappa)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss = -tf.reduce_sum(self.y_true * tf.log(y_pred), axis=1)

        self.grad = tf.gradients(self.loss, self.x_input)[0]

    def perturb(self, sess, x_nat, y, batch_size):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        nb_batch = len(x) // batch_size
        # check if need one more batch
        if nb_batch * batch_size < len(x):
            nb_batch += 1
        
        for i in range(nb_batch):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = np.minimum(end, len(x))
            batch_x = x[start:end]
            batch_y = y[start:end]
            for j in range(self.nb_iter):
                loss, grad = sess.run([self.loss, self.grad],
                                      feed_dict={self.x_input: batch_x,
                                                 self.y_true: batch_y})
                batch_x += self.eps_iter * np.sign(grad)
                batch_x = np.clip(batch_x, x_nat[start:end] - self.epsilon, x_nat[start:end] + self.epsilon)
                batch_x = np.clip(batch_x, self.clip_min, self.clip_max)  # ensure valid pixel range

            x[start:end] = batch_x[:]

        return x

