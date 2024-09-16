# House_Prices_Prediction

## Overview
This project aims to predict house prices using various features such as the number of rooms, square footage, neighborhood, and other factors. We are using the **House Prices: Advanced Regression Techniques** dataset from Kaggle. The goal is to build a regression model that can accurately predict the selling price of a house.

## Dataset
The dataset is taken from the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) competition on Kaggle. The training data contains 1,460 houses, each described by 80 features, including information like:
- Lot size
- Year built
- Overall quality of the house
- Number of rooms
- Garage size
- Neighborhood

The target variable is the house sale price (`SalePrice`).

## Project Structure
The repository is organized as follows:

```
house-price-prediction/
│
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Testing dataset
│
├── notebooks/
│   ├── data_cleaning.ipynb     # Jupyter notebook for data cleaning
│   ├── eda.ipynb               # Jupyter notebook for exploratory data analysis (EDA)
│   ├── model_building.ipynb    # Jupyter notebook for model training and evaluation
│
├── models/
│   ├── house_price_model.pkl   # Trained Random Forest model
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview and details
```

## Project Workflow

### 1. Data Preprocessing
The dataset has missing values and irrelevant features that were cleaned and handled. Some key steps include:
- Handling missing values (filling in with the mean, median, or mode)
- Dropping irrelevant columns (e.g., `PoolQC`, `Fence`, etc.)
- Encoding categorical variables using one-hot encoding

### 2. Exploratory Data Analysis (EDA)
Using visualizations (correlation heatmaps, scatter plots, histograms), patterns and trends were identified to help understand how various features relate to house prices.

### 3. Feature Engineering
Created new features based on the raw data, including transformations and normalizations to make the model more robust.

### 4. Model Building
A **Random Forest Regressor** was used as the primary model, which was trained on the processed dataset. Other models like Linear Regression and Gradient Boosting were also tested.

### 5. Model Evaluation
The model was evaluated using the **Root Mean Squared Error (RMSE)** to assess its performance.

### 6. Prediction
The trained model is used to predict house prices on the test set.

## How to Run the Project

### 1. Clone the repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
```

### 2. Install dependencies
Make sure to have Python 3.x installed. Then, install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place the `train.csv` and `test.csv` files inside the `data/` directory.

### 4. Run the notebooks
Use Jupyter notebooks to run the project. You can start by running:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and open the following notebooks in this order:
- `data_cleaning.ipynb`: For data preprocessing.
- `eda.ipynb`: For data exploration and visualization.
- `model_building.ipynb`: For training the model and making predictions.

### 5. Save the Model
You can train the model and save it using the `joblib` or `pickle` libraries as demonstrated in the notebooks.

## Results
The Random Forest model achieved a **Root Mean Squared Error (RMSE)** of approximately `XXX` on the test set, which demonstrates reasonable accuracy for predicting house prices.

## Dependencies
All the dependencies are listed in the `requirements.txt` file. They include:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

You can install them by running:
```bash
pip install -r requirements.txt
```

## License
This project is open-source and available under the MIT License.
