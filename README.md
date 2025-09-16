# California House Price Prediction with PyTorch
This repository showcases a complete end-to-end supervised learning project for predicting California house prices. It serves as a practical example of a regression task using the `PyTorch` deep learning framework
The entire process, from data preparation to model training and evaluation, is implemented in a single Python script.

## Dataset
The project uses the California Housing Dataset, which contains `20,640` entries with `10` features each. The target variable is `median_house_value`. The dataset includes both numerical features (e.g., `median_income`, `total_rooms`) and a categorical feature (`ocean_proximity`).

## Data Preprocessing
Before training, the raw data must be cleaned and prepared. The following steps were taken:
1. Handling Missing Values: Missing values in the `total_bedrooms` column were filled with the median value of that column.
2. Logarithmic Transformation: Several numerical columns were transformed using `np.log1p` to normalize their skewed distribution, which improves model performance.
3. Outlier Clipping: Outliers in the transformed columns were clipped to the `2nd and 98th percentiles` to prevent extreme values from skewing the model's learning.
4. One-Hot Encoding: The categorical `ocean_proximity` column was converted into a numerical format using one-hot encoding.
5. Data Splitting: The dataset was split into an 80% training set and a 20% testing set.
6. Data Scaling: All features were scaled using `StandardScaler` to ensure they have a similar magnitude, which is crucial for efficient neural network training.
7. Tensor Conversion: The final data was converted into `PyTorch tensors` and loaded into `DataLoader` objects for efficient batch processing.

## Model Architecture
A custom neural network was built using PyTorch's `nn.Module` to solve the regression problem. The model consists of three linear layers and two `ReLU` activation functions to introduce non-linearity.
+ Input Layer: Takes 13 features (after one-hot encoding).
+ Hidden Layer 1: 135 neurons with a ReLU activation function.
+ Hidden Layer 2: 73 neurons with a ReLU activation function.
+ Output Layer: A single neuron that predicts the house value.

The model uses the `Adam optimizer` and a `Mean Squared Error (MSE)` loss function, which is ideal for regression tasks.

## Results
The model was trained for `13` epochs. After training, it was evaluated on the unseen test data.
- Final Test Loss (MSE): 0.082
- Final Test RMSE: 0.287

The low RMSE value indicates that the model's predictions are very close to the actual values on the scaled data. When converted back to the original dollar scale, this RMSE translates to an average prediction error of approximately $17,000, a great result for a first model.

Here are the key visualizations of the training process and results:
### Training Loss Curve
The plot of the training loss shows a steady decrease with each epoch, confirming that the model is actively learning and improving.

### Predicted vs. Actual Values
This scatter plot compares the model's predictions to the actual house values. The points cluster closely around the red dashed line, which represents a perfect prediction. This confirms the model's high accuracy on the test data.

## How to Use
The trained model has been saved as a `.pt` file, allowing it to be loaded and used for future predictions without retraining. You can load it using the following code:
```
# First, you must create a new instance of your model class
loaded_model = house_price_regression_neural_network(feature_size)

# Then, you load the saved state dictionary into the new model instance
loaded_model.load_state_dict(torch.load('house_price_model_params.pt'))

# Switch to evaluation mode to use the loaded model
loaded_model.eval()

# Now the loaded_model is ready to make predictions
```
