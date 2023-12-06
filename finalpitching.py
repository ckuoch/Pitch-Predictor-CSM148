# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score, KFold

# Load your dataset
# Example:
df = pd.read_csv('pitching.csv')

X = df.drop(['pitch_type'], axis = 'columns')
y = df.pitch_type
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape)
print(X_test.shape)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Neural Network

# Standardize the data
scaleFactor = StandardScaler()
X_train_array = scaleFactor.fit_transform(X_train)
X_test_array = scaleFactor.transform(X_test)
y_test_array = y_test.to_numpy()

# Convert to PyTorch tensors
X_train_tensor, y_train_tensor = torch.FloatTensor(X_train_array), torch.LongTensor(y_train)
X_test_tensor, y_test_tensor = torch.FloatTensor(X_test_array), torch.LongTensor(y_test_array)

# Number of folds for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# Placeholder for accuracy scores
accuracy_scores = []
# Define your neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


for fold, (train_index, val_index) in enumerate(kf.split(X_train_tensor)):
    X_fold_train, y_fold_train = X_train_tensor[train_index], y_train_tensor[train_index]
    X_fold_val, y_fold_val = X_train_tensor[val_index], y_train_tensor[val_index]

    train_dataset = TensorDataset(X_fold_train, y_fold_train)
    val_dataset = TensorDataset(X_fold_val, y_fold_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize your neural network
    model = Net()

    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Number of epochs
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    # Evaluation on the validation set

    with torch.no_grad():
        model.eval()
        all_preds = []
        all_labels = []

        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_y.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Fold {fold + 1}, Accuracy: {accuracy}")
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy across folds
average_accuracy = sum(accuracy_scores) / num_folds
print(f"Neural Net Cross Validation Score: {average_accuracy}")

# Logistic Regression
lr_model = LogisticRegression(C = 1, max_iter = 1000)
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)
lr_tacc = lr_model.score(X_train, y_train)
print(f'Logistic Regression Training Accuracy: {lr_tacc}')
print(f'Logistic Regression Accuracy: {lr_accuracy}')

# Random Forest
rf_model = RandomForestClassifier(criterion = 'entropy', max_depth=17, n_estimators = 50)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
rf_tacc = rf_model.score(X_train, y_train)
print(f'Random Forest Training Accuracy: {rf_tacc}')
print(f'Random Forest Accuracy: {rf_accuracy}')

# Decision Tree
decision_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 13)
decision_tree.fit(X_train, y_train)
dt_accuracy = decision_tree.score(X_test, y_test)
dt_tacc = decision_tree.score(X_train, y_train)
print(f'Decision Tree Training Accuracy: {dt_tacc}')
print(f'Decision Tree Accuracy: {dt_accuracy}')

# Cross-validation
lr_cross_val_scores = cross_val_score(lr_model, X, y, cv=5)
print (lr_cross_val_scores)
rf_cross_val_scores = cross_val_score(rf_model, X, y, cv=5)
print(rf_cross_val_scores)
dt_cross_val_scores = cross_val_score(decision_tree, X, y, cv=5)
print(dt_cross_val_scores)

lr_avg_cv_score = np.mean(lr_cross_val_scores)
rf_avg_cv_score = np.mean(rf_cross_val_scores)
dt_avg_cv_score = np.mean(dt_cross_val_scores)

print(f'Logistic Regression Average Cross-validation Score: {lr_avg_cv_score}')
print(f'Random Forest Average Cross-validation Score: {rf_avg_cv_score}')
print(f'Decision Tree Average Cross-validation Score: {dt_avg_cv_score}')

