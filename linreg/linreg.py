"""Code for HW1 Problem 3: Linear Regression and Polynomial Features."""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv

OPTS = None

def read_data(filename, degree):
    x_list = []
    y_list = []
    with open(filename) as f:
        for line in f:
            x, y = [float(t) for t in line.strip().split('\t')]
            x_list.append(featurize(x, d=degree))
            y_list.append(y)
    return np.array(x_list), np.array(y_list)

def featurize(x, d=1):
    if d == 1:
        return [1, x]
    else:
        ### BEGIN_SOLUTION 3c
        features = [1]
        for i in range(d):
            features.append(pow(x,i+1))
        return features
        ### END_SOLUTION 3c

def predict(w, X):
    """Return the predictions using weight vector w on inputs X.

    Args:
        - w: Vector of size (d,)
        - X: Matrix of size (m, d)
    Returns:
        - Predictions vector y_pred of size (m,)
    """
    ### BEGIN_SOLUTION 3a
    return X.dot(w);
    ### END_SOLUTION 3a

def train_gradient_descent(X_train, y_train, lr=1e-2, num_iters=400):
    """Train linear regression using gradient descent.

    Args:
        - X_train: Matrix of size (n, d)
        - y_train: Vector os size (n,)
        - lr: Learning rate (You can just use the default)
        - num_iters: Number of iterations of gradient descent to run (You can just use the default)
    Returns:
        - Weight vector w of size (d,)
    """
    N, D = X_train.shape
    ### BEGIN_SOLUTION 3a
    w = np.zeros(D);
    for i in range(num_iters):
        y_preds = predict(w,X_train);
        per_example_gradients = 2 * X_train.T * (y_preds - y_train)  
        gradient = np.mean(per_example_gradients, axis=1)
        w = w - lr * gradient
    ### END_SOLUTION 3a
    return w

def train_normal_equations(X_train, y_train):
    """Train linear regression using the normal equations.

    Args:
        - X_train: Matrix of size (n, d)
        - y_train: Vector os size (n,)
    Returns:
        - Weight vector w of size (d,)
    """
    
    ### BEGIN_SOLUTION 3b
    xTx = np.dot(X_train.T,X_train)
    xTx_inv = np.linalg.pinv(xTx)
    xTy = np.dot(X_train.T, y_train)
    w = np.dot(xTx_inv, xTy)
    return w
    ### END_SOLUTION 3b

def plot_sweep(degrees, train_rmses, dev_rmses):
    plt.clf()
    plt.plot(degrees, train_rmses, color='r', marker='*', linestyle='-', label='train')
    plt.plot(degrees, dev_rmses, color='b', marker='*', linestyle='-', label='dev')
    plt.xlabel('Degree')
    plt.ylabel('RMSE')
    plt.title('Train/Dev Error vs. Degree of predictor')
    plt.legend()
    plt.show()
    plt.savefig('rmse_vs_degree.png')

def plot_predictors(X_dev, y_dev, plot_xs, plot_data):
    plt.clf()
    plt.plot(X_dev[:,1], y_dev, marker='x', linestyle='')
    for y_preds, degree in plot_data:
        plt.plot(plot_xs, y_preds, marker='', linestyle='-', label=f'd={degree}', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Learned Predictors for Different Degrees')
    plt.legend()
    plt.show()
    plt.savefig('predictions.png')

def evaluate(w, X, y, name):
    """Measure and print RMSE of a predictor on a dataset."""
    y_preds = predict(w, X)
    rmse = np.sqrt(np.mean((y_preds - y)**2))
    print('    {} RMSE: {}'.format(name, rmse))
    return rmse

def parse_degree(s):
    if ':' in s:
        start, end = s.split(':')
        return list(range(int(start), int(end) + 1))
    return [int(s)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a', choices=['gradient_descent', 'normal'], required=True)
    parser.add_argument('--degree', '-d', type=parse_degree, default=[1])
    parser.add_argument('--downsample-to', '-n', type=int, default=0)
    parser.add_argument('--plot-degrees', '-p', type=lambda s: [int(x) for x in s.split(',')], default=None)
    parser.add_argument('--test', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main():
    train_rmses = []
    dev_rmses = []
    plot_data = []
    plot_xs = np.linspace(-2, 2, num=100)
    for degree in OPTS.degree:
        print(f'Degree {degree}:')

        # Read in the data
        X_train, y_train = read_data('train.tsv', degree)
        if OPTS.downsample_to:  # -n flag: Only use the first few examples
            X_train = X_train[:OPTS.downsample_to,]
            y_train = y_train[:OPTS.downsample_to]
        X_dev, y_dev = read_data('dev.tsv', degree)
        X_test, y_test = read_data('test.tsv', degree)

        # Train, either with gradient descent or normal equations
        if OPTS.algorithm == 'gradient_descent':
            w = train_gradient_descent(X_train, y_train)
        elif OPTS.algorithm == 'normal':
            w = train_normal_equations(X_train, y_train)

        # Evaluate on train, dev, and (if --test flag) test sets
        train_rmse = evaluate(w, X_train, y_train, 'Train')
        dev_rmse = evaluate(w, X_dev, y_dev, 'Dev')
        train_rmses.append(train_rmse)
        dev_rmses.append(dev_rmse)
        if OPTS.test:
            evaluate(w, X_test, y_test, 'Test')

        # Store data to make plot of the learned functions, if -p flag
        if OPTS.plot_degrees and degree in OPTS.plot_degrees:
            X_plot = np.array([featurize(x, d=degree) for x in plot_xs])
            y_preds = predict(w, X_plot)
            plot_data.append((y_preds, degree))

    # Plot the train and dev RMSE as a function of degree
    if len(OPTS.degree) > 1:
        plot_sweep(OPTS.degree, train_rmses, dev_rmses)

    # Plot the learned functions, if -p flag
    if OPTS.plot_degrees and plot_data:
        X_dev, y_dev = read_data('dev.tsv', 1)
        plot_predictors(X_dev, y_dev, plot_xs, plot_data)

if __name__ == '__main__':
    OPTS = parse_args()
    main()

