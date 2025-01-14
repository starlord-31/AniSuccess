<div align="center">
  <h1>AniSuccess</h1>
  <img src="https://images3.alphacoders.com/132/thumb-1920-1323165.png" alt="Anime Theme">
</div>

The **AniSuccess** project uses machine learning to predict whether an anime will be successful based on features like genre, studio, rating, and more. Success is defined by factors such as high ratings, popularity, and audience engagement.

This tool is designed for anime studios, streaming platforms, and fans to understand what drives success in anime. It includes:

- **Data Preparation**: Cleaning and preprocessing anime metadata.
- **Model Training**: Training a predictive model using XGBoost.
- **API Deployment**: Deploying the model as an API with Flask for real-time predictions.
- **Scalable Deployment**: Using Docker and AWS Elastic Beanstalk for easy deployment and scalability.

This project helps make data-driven decisions about anime production and marketing.

---

## Table of Contents

1. [Data Collection](#data-collection)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Model Training](#model-training)
4. [Model Deployment](#model-deployment)
5. [How to Run Locally](#how-to-run-locally)
6. [Using Docker](#using-docker)

---

## Data Collection

The dataset was collected using the [Jikan API](https://jikan.moe/), which provides metadata from MyAnimeList. The data includes features like:

- Title
- Genre
- Studio
- Rating
- Popularity
- Number of episodes
- Synopsis

The data collection process is documented in this notebook:  
[Data Collection Notebook](https://github.com/starlord-31/AniSuccess/blob/main/Data%20Collection.ipynb)

---

## Exploratory Data Analysis (EDA)

Key steps of EDA included:

1. **Handling Missing Values**:
   - Filled missing genres, studios, and ratings.
   - Created new features like `synopsis_length` and `duration_minutes`.

2. **Feature Engineering**:
   - Defined the target variable `success` based on popularity, ratings, and audience engagement.

3. **Visualizations**:
   - Analyzed distributions and correlations between features.

The complete EDA process is documented here:  
[EDA and Data Modeling Notebook](https://github.com/starlord-31/AniSuccess/blob/main/EDA%20and%20Data%20Modeling.ipynb)

---

## Model Training

The predictive model was trained using **XGBoost** with the following steps:

1. **Feature Transformation**:
   - Used `DictVectorizer` to handle categorical features.
   
2. **Cross-Validation**:
   - Performed K-Fold Cross-Validation to evaluate the model.

3. **Hyperparameter Tuning**:
   - Adjusted parameters like learning rate and maximum depth.

The training process can be found in this script:  
[train.py](https://github.com/starlord-31/AniSuccess/blob/main/train.py)

---

## Model Deployment

The trained model is deployed as a REST API using **Flask** and **Gunicorn** for real-time predictions. The deployment includes:

- **Saving the Trained Model**: Stored in `model_C=1.0.bin`.
- **Running the API Locally**: Tested with a sample input.

The deployment code can be found here:  
[predict.py](https://github.com/starlord-31/AniSuccess/blob/main/predict.py)  
Test the API locally using this script:  
[predict_test.py](https://github.com/starlord-31/AniSuccess/blob/main/predict_test.py)

---

## How to Run Locally

### Prerequisites
- Python 3.8 or higher
- Pipenv
- Docker (optional)

### Steps to Run Locally

1. **Clone the Repository**
   ```bash
   git clone git@github.com:starlord-31/AniSuccess.git
   cd AniSuccess

