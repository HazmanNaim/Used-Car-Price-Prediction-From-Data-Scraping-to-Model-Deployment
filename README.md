# Project Title: "Predicting Used Car Prices in the Malaysian Market"

Libraries and Packages:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![Postman](https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=postman&logoColor=white)

### Introduction
This documentation provides a comprehensive overview of the project titled "Predicting Used Car Prices in the Malaysian Market." The project involves using machine learning techniques to predict the prices of used cars in the Malaysian market based on a dataset collected from web scraping. This documentation will cover various aspects of the project, including its objectives, methodology, challenges, and future goals.

### Project Objectives
The primary objective of this project is to develop a machine learning model that can accurately predict the prices of used cars in the Malaysian market. To achieve this, the following specific objectives were set:

1. Collect a substantial dataset of used car listings in the Malaysian market through web scraping.
2. Perform data cleaning and preprocessing to prepare the dataset for modeling.
3. Develop a machine learning pipeline that includes one-hot encoding, feature scaling, and a random forest regressor model.
4. Train the model on the dataset and optimize it using cross-validation and hyperparameter tuning.
5. Deploy the trained model as a web application using Flask, HTML, and CSS.

### Determinants of Used Car Prices
One of the key questions addressed in this project is, "What determines most of the used car prices in the Malaysian market?" The answer to this question is essential for understanding the factors that influence the pricing of used cars. The machine learning model developed in this project aims to identify and quantify these determinants based on the dataset.

### Project Flow
The project follows a structured workflow consisting of the following steps:

1. **Web Scraping**: A total of 8000 listings of used car prices in the Malaysian market were collected through web scraping. Each listing contains valuable information about the cars, such as brand, engine cc, year, mileage, and price.

2. **Data Cleaning**: The collected data underwent extensive data cleaning, which involved data imputation and feature engineering. This step was crucial for preparing the dataset for model development.

3. **Model Development**: A machine learning pipeline was created, which included data transformation (one-hot encoding and feature scaling) and a random forest regressor model. The model was trained on the cleaned dataset to predict the prices of used cars. Cross-validation with 10 folds and hyperparameter tuning were performed to optimize model performance. The model achieved a cross-validated Root Mean Square Error (RMSE) of 27458.34 and a cross-validated R-squared (R2) of 0.927090. This ensured that the model neither underfits nor overfits the data.

4. **Model Deployment**: The trained model was deployed as a web application using Flask as the primary web framework and HTML & CSS for the frontend. This deployment allows users to input car details and receive price predictions.

### Key Learnings
Throughout the course of this project, several valuable lessons were learned:

- **Data Collection**: The process of web scraping can be slow and prone to interruptions, leading to challenges in collecting a large dataset.
  
- **Data Cleaning**: Data cleaning and preprocessing are time-consuming but essential steps in ensuring the quality of the dataset and the accuracy of the model.

- **Web Development**: Developing a web application, especially for those with limited domain knowledge in web development, can be a challenging but rewarding learning experience.

### Challenges Faced
The project presented its fair share of challenges:

- **Slow Web Scraping**: Web scraping was slower than anticipated, and interruptions occurred, limiting the dataset to 8000 listings.

- **Data Cleaning Complexity**: Cleaning and preprocessing the data consumed a significant portion of the project's time due to the diverse nature of the data.

- **Limited Web Development Knowledge**: Developing a web app required self-learning, especially for those not well-versed in web development technologies.

### Future Goals
While the project has achieved its primary objectives, there are several future goals and improvements that can be considered:

- **Data Collection**: Efforts can be made to collect a larger dataset to further improve model accuracy.

- **Cloud Deployment**: Deploying the web app to a cloud hosting platform would make it accessible from anywhere and enhance its scalability.

- **Feature Engineering**: Further exploration of feature engineering techniques may enhance the model's predictive power by capturing additional nuances in used car pricing.

### Conclusion
The project "Predicting Used Car Prices in the Malaysian Market" successfully developed a machine learning model that can predict the prices of used cars based on a dataset collected through web scraping. Despite challenges in data collection, cleaning, and web development, the project achieved its objectives and provides a foundation for future improvements and enhancements. Understanding the determinants of used car prices in the Malaysian market is valuable for both buyers and sellers in making informed decisions.

### Additional Notes
- During the model deployment to the web app, it's worth noting that only half of the features were chosen for deployment compared to what was used during model training. This selective feature deployment was done as the primary goal of the model deployment phase was to practice skills and knowledge related to deploying machine learning models using Flask. Similarly, the web app's design was kept simple, as the primary focus was to demonstrate the model deployment process rather than creating an elaborate user interface.
