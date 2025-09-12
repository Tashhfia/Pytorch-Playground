import torch
import numpy as np
import pathlib as Path
import os
import matplotlib.pyplot as plt

# def explore_dir(dir_path):

#     for file in 

def visualize_linear(X_train, Y_train, X_test, Y_test, Pred=None):
    plt.scatter(X_train.numpy(), Y_train.numpy(), label = "Training Data", c="b")
    plt.scatter(X_test.numpy(), Y_test.numpy(), label = "Test Data", c="r")
    if Pred is not None:
    # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(X_test.numpy(), Pred.numpy(), c="r", s=4, label="Predictions")
    plt.title("Training and Testing Data Visualized")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())