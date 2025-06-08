#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        """
        # Compute the scores for each class
        scores = np.dot(self.W, x_i)  # (n_classes)
        # Predict the label
        y_hat = scores.argmax()
        # Update weights only if the prediction is incorrect
        if y_hat != y_i:
            # Increase weights for the correct class
            self.W[y_i] += x_i
            # Decrease weights for the predicted (incorrect) class
            self.W[y_hat] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): learning rate for weight updates
        l2_penalty (float): strength of L2 regularization
        """
        # Compute the linear scores for each class
        scores = np.dot(self.W, x_i)  # Shape: (n_classes,)
        
        # Apply the softmax function to get predicted probabilities
        exp_scores = np.exp(scores - np.max(scores))  # For numerical stability
        probabilities = exp_scores / np.sum(exp_scores)  # Shape: (n_classes,)
        
        # Create the true distribution (one-hot encoding)
        y_true = np.zeros_like(probabilities)
        y_true[y_i] = 1
        
        # Compute the gradient of the loss w.r.t. weights (including L2 regularization)
        gradient = np.outer((probabilities - y_true), x_i)  # Shape: (n_classes, n_features)
        
        # Add L2 regularization term to the gradient
        gradient += l2_penalty * self.W  # L2 regularization term
        
        # Update the weights using the gradient and learning rate
        self.W -= learning_rate * gradient


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize weights and biases
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        # Weights from input to hidden layer
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features))
        # Biases for hidden layer
        self.b1 = np.zeros(hidden_size)

        # Weights from hidden to output layer
        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size))
        # Biases for output layer
        self.b2 = np.zeros(n_classes)

    def predict(self, X):
        # Forward pass
        # Hidden layer
        hidden_input = np.dot(X, self.W1.T) + self.b1  # Shape: (n_examples, hidden_size)
        hidden_output = np.maximum(0, hidden_input)     # ReLU activation

        # Output layer
        output_scores = np.dot(hidden_output, self.W2.T) + self.b2  # Shape: (n_examples, n_classes)

        # Predicted labels
        y_pred = np.argmax(output_scores, axis=1)
        return y_pred

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = np.sum(y == y_hat)
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Train the model for one epoch using stochastic gradient descent.
        Don't forget to return the loss of the epoch.
        """
        n_examples = X.shape[0]
        total_loss = 0.0

        for i in range(n_examples):
            x_i = X[i]  # Shape: (n_features,)
            y_i = y[i]  # Scalar

            # Forward pass
            # Hidden layer
            z1 = np.dot(self.W1, x_i) + self.b1        # Shape: (hidden_size,)
            a1 = np.maximum(0, z1)                     # ReLU activation

            # Output layer
            z2 = np.dot(self.W2, a1) + self.b2         # Shape: (n_classes,)

            # Compute probabilities (softmax)
            exp_scores = np.exp(z2 - np.max(z2))       # For numerical stability
            probs = exp_scores / np.sum(exp_scores)    # Shape: (n_classes,)

            # Compute cross-entropy loss
            correct_logprob = -np.log(probs[y_i])
            total_loss += correct_logprob

            # Backpropagation
            # Gradient on scores
            dz2 = probs
            dz2[y_i] -= 1                              # Shape: (n_classes,)

            # Gradients for W2 and b2
            dW2 = np.outer(dz2, a1)                    # Shape: (n_classes, hidden_size)
            db2 = dz2                                   # Shape: (n_classes,)

            # Backprop into hidden layer
            da1 = np.dot(self.W2.T, dz2)               # Shape: (hidden_size,)
            dz1 = da1 * (z1 > 0)                       # ReLU derivative

            # Gradients for W1 and b1
            dW1 = np.outer(dz1, x_i)                   # Shape: (hidden_size, n_features)
            db1 = dz1                                   # Shape: (hidden_size,)

            # Parameter update
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        # Return average loss over the epoch
        avg_loss = total_loss / n_examples
        return avg_loss


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
        print(opt.hidden_size, opt.epochs)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
