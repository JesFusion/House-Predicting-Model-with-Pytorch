import pandas as pd
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time




# =========== LOADING/GATHERING DATA ==============
practice_dataset = pd.read_csv(r"C:\Users\USER\Documents\My Programs\Machine_Learning\Python Programs\jesseVirtualEnvs\.mainDog\python code\data_sets\California_Housing_Dataset.csv")


# ============ ANALYZING DATA =================

# print(practice_dataset.info())
# we're having missing values in this dataset



# ============= REMOVING DUPLICATE SAMPLES ===================

# removing duplicate samples(ie, rows), not cells

practice_dataset.drop_duplicates(inplace = True)


# =============== FILLING/HANDLING MISSING VALUES ===========

# we'll be filling missing values in each column with median values

total_bedrooms_MV = practice_dataset["total_bedrooms"].median()

practice_dataset["total_bedrooms"].fillna(total_bedrooms_MV, inplace = True)



# =============== APPLYING LOGRARITHMIC TRANSFORMATION AND CLIPPING OUTLIERS =====================

# most columns have a skewed distribution. so we'll apply logarithmic transformation to normalize them, perfect for our linear regression model

for a_column in practice_dataset.columns.tolist()[3:9]: #['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

    # applying logarithmic transformation...
    practice_dataset[a_column] = np.log1p(practice_dataset[a_column])

    # removing outliers by clipping...
    column_upper_boundary = practice_dataset[a_column].quantile(0.99)

    column_lower_boundary = practice_dataset[a_column].quantile(0.01)

    practice_dataset[a_column] = practice_dataset[a_column].clip(lower = column_lower_boundary, upper = column_upper_boundary)




# =================== CONVERTING CATEGORICAL DATA TO NUMERICAL THROUGH ONE HOT ENCODING ====================

practice_dataset = pd.get_dummies(practice_dataset, columns = ["ocean_proximity"], dtype = int)

time.sleep(2.4)

print("Categorical column \"ocean_proximity\" converted to numerical successfully...")



# ================= SPLITTING DATA =================

x_sect = practice_dataset.drop("median_house_value", axis = 1)

y_sect = practice_dataset["median_house_value"]

# print(x_sect.shape, y_sect.shape)

x_train, x_test, y_train, y_test = train_test_split(x_sect, y_sect, random_state = 19, test_size = 0.26)


# ================= SCALING THE DATA TO IMPROVE MODEL PERFORMANCE ==================

scaling_model = StandardScaler()

x_train = scaling_model.fit_transform(x_train)

x_test = scaling_model.transform(x_test)



# =================== CONVERTING TO PYTORCH TENSORS ===============


# converting dataset to tensors for efficient feeding to model
x_train = th.tensor(x_train, dtype = th.float32)

y_train = th.tensor(y_train.values, dtype = th.float32).unsqueeze(1)

x_test = th.tensor(x_test, dtype = th.float32)

y_test = th.tensor(y_test.values, dtype = th.float32).unsqueeze(1)





# ================= CREATING DATASET AND DATALOADER ====================

class dataset_on_houses(Dataset):
    def __init__(self, features, targets):
        self.features = features

        self.targets = targets

    
    def __len__(self):
        return len(self.features)
    

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
    

dataset_for_training = dataset_on_houses(x_train, y_train)

dataset_for_testing = dataset_on_houses(x_test, y_test)


# creating dataloader objects...

train_dataset_loader = DataLoader(
    dataset = dataset_for_training,
    batch_size = 59,
    shuffle = True
)

test_dataset_loader = DataLoader(
    dataset = dataset_for_testing,
    batch_size = 59,
    shuffle = False
)



# ================= BUILDING THE NEURAL NETWORK =====================

feature_size = x_train.shape[1]

class house_price_regression_neural_network(nn.Module):
    def __init__(self, input_size):
        super(house_price_regression_neural_network, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_size, 135),

            nn.ReLU(),

            nn.Linear(135, 73),

            nn.ReLU(),

            nn.Linear(73, 1)
        )

    
    def forward(self, x):
        return self.network(x)
    


ai_algorithm = house_price_regression_neural_network(feature_size)

print("Model Instantiated...")

time.sleep(2.4)




# ================== DEFINE LOSS FUNCTION AND OPTIMIZER ==================

lr, e = 0.0012, 14

loss_function = nn.MSELoss()

model_optimizer = optim.Adam(ai_algorithm.parameters(), lr = lr)



# ================= TRAINING THE MODEL ==================

list_of_losses = []


print(f"Starting training for {e} epochs...\n")

time.sleep(2.3)

for an_epoch in range(e):
    for independent_variable, dependent_variable in train_dataset_loader:
        ai_model_prediction = ai_algorithm(independent_variable)


        ai_model_losses = loss_function(ai_model_prediction, dependent_variable)

        model_optimizer.zero_grad()

        ai_model_losses.backward()

        model_optimizer.step()

    
    list_of_losses.append(ai_model_losses.item())

    print(f"Epoch --> [{an_epoch + 1}/{e}]\nLoss --> {ai_model_losses.item():.3f}\n")


print("Training model complete...")

time.sleep(2.1)



# ================== EVALUATING THE MODEL ===================

ai_algorithm.eval()

losses_in_test = 0

predictions_list = []

targets_list = []


with th.no_grad():
    for features, targets in test_dataset_loader:
        ai_prediction = ai_algorithm(features)

        losses_in_test += loss_function(ai_prediction, targets).item()

        predictions_list.extend(ai_prediction.squeeze().tolist())

        targets_list.extend(targets.squeeze().tolist())



test_loss_average = losses_in_test / len(test_dataset_loader)

print(f"\nAverage Test Loss --> {test_loss_average:.3f}")

print(f"\nAverage RSME Loss --> {test_loss_average**0.5:.3f}")





# =================== VISUALIZING REAULTS ====================


plt.figure(figsize = (10, 5))

plt.plot(list_of_losses, label = "Loss during Training", c = "orange")

plt.title("Training Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()



# plotting predicted vs. acutal values


plt.figure(figsize = (10, 10))

plt.scatter(targets_list, predictions_list, alpha = 0.5)

plt.title("Predicted vs. Actual House Values")

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.plot(
    [min(targets_list), max(targets_list)],

    [min(targets_list), max(targets_list)], "r--"
)

plt.show()






# ===================== SAVING THE MODEL ======================

th.save(ai_algorithm.state_dict(), "house_price_regression_predicting_model_paramaters.pt")


