Alphabet Soup Charity Fund Prediction

Overview

Alphabet Soup, a nonprofit organization, seeks to enhance its funding process by predicting which applicants have the best chance of success if funded. This project utilizes machine learning and neural networks to create a binary classification model that can predict the success of funding applicants based on a variety of features.

Project Structure

1. Data Preprocessing
   
The data used in this project consists of over 34,000 organizations that have received funding from Alphabet Soup over the years. The dataset includes several columns that capture metadata about each organization, such as:

EIN and NAME: Identification columns (removed during preprocessing).
APPLICATION_TYPE: Alphabet Soup application type.
AFFILIATION: Affiliated sector of industry.
CLASSIFICATION: Government organization classification.
USE_CASE: Use case for funding.
ORGANIZATION: Organization type.
STATUS: Active status.
INCOME_AMT: Income classification.
SPECIAL_CONSIDERATIONS: Special considerations for application.
ASK_AMT: Funding amount requested.
IS_SUCCESSFUL: Target variable indicating whether the money was used effectively.
Preprocessing Steps
Target Variable: IS_SUCCESSFUL
Features: All columns except EIN and NAME.

Dropping Irrelevant Columns: The EIN and NAME columns were dropped.

Categorical Encoding: All categorical variables were encoded using pd.get_dummies() to ensure they could be used in the model.

Combining Rare Categories: For columns with many unique values, rare categories were combined into an "Other" category to reduce noise.

Data Splitting: The data was split into features (X) and target (y) arrays, and further split into training and testing datasets using train_test_split.

Scaling: The features were normalized using StandardScaler to ensure the model could process the data effectively.

2. Model Creation
Using TensorFlow and Keras, a neural network was designed to classify the success of funding applicants. The network's architecture was chosen based on the number of input features, with experimentation in layer and neuron counts to optimize performance.

Model Architecture

Hidden Layers: The model includes multiple hidden layers with different activation functions (e.g., ReLU) to capture complex patterns in the data.

Output Layer: The output layer uses a Sigmoid activation function to predict the binary outcome (IS_SUCCESSFUL).

Compilation and Training: The model was compiled and trained using appropriate loss functions and optimizers, with a callback to save the model's weights every five epochs.

Evaluation: The model was evaluated using the test data, and the results were saved in an HDF5 file named AlphabetSoupCharity.h5.

3. Model Optimization
To improve the model's accuracy, several optimization techniques were employed:

Adjusting Model Architecture: Additional neurons and layers were added, with experimentation in activation functions.

Hyperparameter Tuning: Different learning rates, optimizers (like Adam and SGD), and batch sizes were tested.

Training Regimen: The number of epochs was adjusted to find the optimal training duration.

The optimized model was saved in an HDF5 file named AlphabetSoupCharity_Optimization.h5.

4. Results and Analysis
The model was evaluated based on its ability to predict the success of funding applicants. Key findings include:

Performance: The model's accuracy and loss metrics were used to gauge its effectiveness, with efforts to achieve a target accuracy above 75%.

Challenges: Challenges encountered during the project, such as dealing with imbalanced data and optimizing hyperparameters, were addressed with various techniques.

Explore the Results:

The trained model is saved as AlphabetSoupCharity.h5 and AlphabetSoupCharity_Optimization.h5 for the optimized version. You can load these models for further analysis or predictions.


Acknowledgments
Alphabet Soup: For providing the dataset and project scope.
Kaggle and TensorFlow: For the tools and resources used in this project.
FP Central Grading Team: For valuable feedback to improve the project.
