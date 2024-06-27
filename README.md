# Assignment-2
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
np.set_printoptions(suppress=True) # suppress scientific notation

my_file = Path("./Housing.csv")#Reads the file path
if not my_file.is_file():
    print("Housing.csv does not exist.") #Check 1
lines = [line.strip() for line in open("./Housing.csv")]
cols = lines[0].split(',')
train_len = int(0.9*len(lines)-1)
del lines[0] # delete column header so it doesnt get shuffled into the mix

def generate_data(normalize=False, standardize=False):
    train_x = {col:[] for col in cols}
    test_x = {col:[] for col in cols}
    assert not (normalize and standardize), "you cannot normalize and standardize"

    random.shuffle(lines) # unsorts

    # training data (80%)
    for line in lines[1:train_len]:
        line = line.split(',')
        for i, col in enumerate(cols):
            train_x[col].append(line[i])
    # validation data (20%)
    for line in lines[train_len+1:]:
        line = line.split(',')
        for i, col in enumerate(cols):
            test_x[col].append(line[i])

    #formulas
    norm = lambda col: [(x - min(col)) / (max(col) - min(col)) for x in col]
    std = lambda col: [(x - sum(col) / len(col)) / (max(col) - min(col)) for x in col]

    def process(df):
        for key in df.keys():
            # these are yes and no columns
            if key in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
                df[key] = list(map(lambda val: 1 if val == "yes" else 0, df[key]))
            # this one is furnished, semi-... etc
            elif key == "furnishingstatus":
                df[key] = list(map(lambda val: 1 if val == "furnished" else 0.5 if val == "semi-furnished" else 0, df[key]))
            #Converted nums
            else:
                df[key] = list(map(int, df[key]))

            if normalize:
                df[key] = norm(df[key])
            if standardize:
                df[key] = std(df[key])

    process(train_x)
    process(test_x)

    train_y = train_x["price"]; del train_x["price"]
    test_y = test_x["price"]; del test_x["price"]

    return train_x, np.array(train_y), test_x, np.array(test_y)

import numpy as np
import matplotlib.pyplot as plt

# Function to plot model parameters
def plot_parameters(cols, w):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cols, w, color='skyblue', edgecolor='navy')
    plt.title('Model Parameters', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Parameter Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Adding parameter values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}', ha='center', va='bottom')

    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()

# Function to plot training and validation losses
def plot_losses(losses, val_losses, epochs):
    plt.figure(figsize=(10, 6))
    xaxis = np.arange(1, epochs + 1)
    plt.plot(xaxis, losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(xaxis, val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Training function using gradient descent
def train(w, b, xtrain, ytrain, xtest, ytest, epochs, lr, l1_reg=None):
    losses = []
    val_losses = []

    for i in range(epochs):
        # Forward pass: Predicting the output
        outs = xtrain.dot(w) + b

        # Calculating training loss (mean squared error)
        loss = np.mean((outs - ytrain) ** 2)

        # Adding L1 regularization penalty to the loss
        if l1_reg is not None:
            loss += l1_reg * np.sum(np.abs(w))
        losses.append(loss)

        # Backward pass: Calculating gradients
        m = xtrain.shape[0]
        dw = (2/m) * np.dot(xtrain.T, (outs - ytrain)) + (l1_reg * np.sign(w) if l1_reg is not None else 0)
        db = (2/m) * np.sum(outs - ytrain)

        # Updating parameters
        w -= lr * dw
        b -= lr * db

        # Calculating validation loss
        val_outs = xtest.dot(w) + b
        val_loss = np.mean((ytest - val_outs) ** 2)
        val_losses.append(val_loss)

    return losses, val_losses

# Function to generate data (stub for actual data generation)
def generate_data(normalize=False):
    # Replace this with actual data loading and preprocessing logic
    # Dummy data for illustration
    np.random.seed(0)
    xtrain = np.random.rand(100, 5)
    ytrain = np.random.rand(100)
    xtest = np.random.rand(20, 5)
    ytest = np.random.rand(20)
    return xtrain, ytrain, xtest, ytest

# Part 1a: Predicting housing price without normalization or standardization
xtrain, ytrain, xtest, ytest = generate_data()
included_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
w = np.random.randn(xtrain.shape[1]) / 10
b1 = 0
epochs = 1000
losses, val_losses = train(w, b1, xtrain, ytrain, xtest, ytest, epochs, lr=0.01)
print(losses[-1])

# Part 1b: Same as 1a with more parameters
xtrain, ytrain, xtest, ytest = generate_data()
included_cols = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea"]
w = np.random.randn(xtrain.shape[1]) / 10
b1 = 0
losses, val_losses = train(w, b1, xtrain, ytrain, xtest, ytest, epochs, lr=0.01)
print(losses[-1])

# Part 2a: With normalization
xtrain, ytrain, xtest, ytest = generate_data(normalize=True)
included_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
w = np.random.randn(xtrain.shape[1]) / 10
b1 = 0
losses, val_losses = train(w, b1, xtrain, ytrain, xtest, ytest, epochs, lr=0.01)
plot_losses(losses, val_losses, epochs)

# Part 2b: With normalization and more parameters
xtrain, ytrain, xtest, ytest = generate_data(normalize=True)
included_cols = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea"]
w = np.random.randn(xtrain.shape[1]) / 10
b1 = 0
losses, val_losses = train(w, b1, xtrain, ytrain, xtest, ytest, epochs, lr=0.01)
plot_losses(losses, val_losses, epochs)

# Part 3a: With L1 regularization
xtrain, ytrain, xtest, ytest = generate_data(normalize=True)
included_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
w = np.random.randn(xtrain.shape[1]) / 10
b1 = 0
losses, val_losses = train(w, b1, xtrain, ytrain, xtest, ytest, epochs, lr=0.01, l1_reg=0.01)
plot_losses(losses, val_losses, epochs)
plot_parameters(included_cols, w)

# Part 3b: With L1 regularization and more parameters
xtrain, ytrain, xtest, ytest = generate_data(normalize=True)
included_cols = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea"]
w = np.random.randn(xtrain.shape[1]) / 10
b1 = 0
losses, val_losses = train(w, b1, xtrain, ytrain, xtest, ytest, epochs, lr=0.01, l1_reg=0.01)
plot_losses(losses, val_losses, epochs)
plot_parameters(included_cols, w)
