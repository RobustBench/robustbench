import tensorflow as tf
import numpy as np
import torch

class ModelAdapter():
    def __init__(self, logits, x, y, sess, num_classes=10):
        self.logits = logits
        self.sess = sess
        self.x_input = x
        self.y_input = y
        self.num_classes = num_classes
        
        # gradients of logits
        if num_classes <= 10:
            self.grads = [None] * num_classes
            for cl in range(num_classes):
                self.grads[cl] = tf.gradients(self.logits[:, cl], self.x_input)[0]
        
        # cross-entropy loss
        self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.y_input)
        self.grad_xent = tf.gradients(self.xent, self.x_input)[0]
        
        # dlr loss
        self.dlr = dlr_loss(self.logits, self.y_input, num_classes=self.num_classes)
        self.grad_dlr = tf.gradients(self.dlr, self.x_input)[0]
        
        # targeted dlr loss
        self.y_target = tf.placeholder(tf.int64, shape=[None])
        self.dlr_target = dlr_loss_targeted(self.logits, self.y_input, self.y_target, num_classes=self.num_classes)
        self.grad_target = tf.gradients(self.dlr_target, self.x_input)[0]
    
    def predict(self, x):
        x2 = np.moveaxis(x.cpu().numpy(), 1, 3)
        y = self.sess.run(self.logits, {self.x_input: x2})
        
        return torch.from_numpy(y).cuda()

    def grad_logits(self, x):
        x2 = np.moveaxis(x.cpu().numpy(), 1, 3)
        g2 = self.sess.run(self.grads, {self.x_input: x2})
        g2 = np.moveaxis(np.array(g2), 0, 1)
        g2 = np.transpose(g2, (0, 1, 4, 2, 3))
        
        return torch.from_numpy(g2).cuda()

    def set_target_class(self, y, y_target):
        la = y.cpu().numpy()
        la_target = y_target.cpu().numpy()
        l = [None] * y.shape[0]
        for c in range(y.shape[0]):
            l[c] = self.logits[c, la_target[c]] - self.logits[c, la[c]]
        self.diff_logits = tf.stack(l)
        self.grad_diff_logits = tf.gradients(self.diff_logits, self.x_input)[0]
    
    def get_grad_diff_logits_target(self, x, y=None, y_target=None):
        x2 = np.moveaxis(x.cpu().numpy(), 1, 3)
        dl, g2 = self.sess.run([self.diff_logits, self.grad_diff_logits], {self.x_input: x2})
        g2 = np.transpose(np.array(g2), (0, 3, 1, 2))
        
        return torch.from_numpy(dl).cuda(), torch.from_numpy(g2).cuda()
    
    def get_logits_loss_grad_xent(self, x, y):
        x2 = np.moveaxis(x.cpu().numpy(), 1, 3)
        y2 = y.clone().cpu().numpy()
        logits_val, loss_indiv_val, grad_val = self.sess.run([self.logits, self.xent, self.grad_xent], {self.x_input: x2, self.y_input: y2})
        grad_val = np.moveaxis(grad_val, 3, 1)
        
        return torch.from_numpy(logits_val).cuda(), torch.from_numpy(loss_indiv_val).cuda(), torch.from_numpy(grad_val).cuda()

    def get_logits_loss_grad_dlr(self, x, y):
        x2 = np.moveaxis(x.cpu().numpy(), 1, 3)
        y2 = y.clone().cpu().numpy()
        logits_val, loss_indiv_val, grad_val = self.sess.run([self.logits, self.dlr, self.grad_dlr], {self.x_input: x2, self.y_input: y2})
        grad_val = np.moveaxis(grad_val, 3, 1)
        
        return torch.from_numpy(logits_val).cuda(), torch.from_numpy(loss_indiv_val).cuda(), torch.from_numpy(grad_val).cuda()
    
    def get_logits_loss_grad_target(self, x, y, y_target):
        x2 = np.moveaxis(x.cpu().numpy(), 1, 3)
        y2 = y.clone().cpu().numpy()
        y_targ = y_target.clone().cpu().numpy()
        logits_val, loss_indiv_val, grad_val = self.sess.run([self.logits, self.dlr_target, self.grad_target], {self.x_input: x2, self.y_input: y2, self.y_target: y_targ})
        grad_val = np.moveaxis(grad_val, 3, 1)
        
        return torch.from_numpy(logits_val).cuda(), torch.from_numpy(loss_indiv_val).cuda(), torch.from_numpy(grad_val).cuda()

def dlr_loss(x, y, num_classes=10):
    x_sort = tf.contrib.framework.sort(x, axis=1)
    y_onehot = tf.one_hot(y, num_classes)
    ### TODO: adapt to the case when the point is already misclassified
    loss = -(x_sort[:, -1] - x_sort[:, -2]) / (x_sort[:, -1] - x_sort[:, -3] + 1e-12)

    return loss

def dlr_loss_targeted(x, y, y_target, num_classes=10):
    x_sort = tf.contrib.framework.sort(x, axis=1)
    y_onehot = tf.one_hot(y, num_classes)
    y_target_onehot = tf.one_hot(y_target, num_classes)
    loss = -(tf.reduce_sum(x * y_onehot, axis=1) - tf.reduce_sum(x * y_target_onehot, axis=1)) / (x_sort[:, -1] - .5 * x_sort[:, -3] - .5 * x_sort[:, -4] + 1e-12)
    
    return loss