# Predicting-the-City_Cycle-in-Fuel-Consumption
## Background Information

### Data Files
The primary dataset, mpg.data.xlsx, contains information on eight car attributes, including MPG, cylinders, horsepower, and car name. These data help car rental companies make informed decisions about updating their fleets by predicting fuel efficiency (MPG).

### Resources
The dataset is provided under the Creative Commons CC0 1.0 Universal (CC0 1.0) 'Public Domain Dedication' license ( https://creativecommons.org/publicdomain/zero/1.0/ ) , so we are authorized to use it in this project.

### Data Description
We were provided with the Auto MPG dataset, which includes the following characteristics:

![image](https://github.com/user-attachments/assets/9e47cf20-def0-43d5-b60e-475380100b5b)

Our table contains eight attributes, as shown below:

![image](https://github.com/user-attachments/assets/792c14c7-0577-4f7d-b6ae-13a2b1d251b4)

### Data Processes
#### Exploratory Data Analysis
The first step in our analysis was exploring the data. We began by importing the necessary Python libraries to visualize the data and calculate key metrics for our model. Below you can see some of our frequently used commands for that step.

![image](https://github.com/user-attachments/assets/44edf878-c6ed-48af-b91d-74f1c84c5711)

![image](https://github.com/user-attachments/assets/bfad36f3-0204-42ac-be51-2741745657ae)

After identifying key metrics during the EDA step, we proceeded with adjustments and corrections for erroneous values.

![image](https://github.com/user-attachments/assets/3165264a-36fb-4e0a-a683-128cc1037a3d)

![image](https://github.com/user-attachments/assets/61e359f7-495d-4cad-96fa-098c12d63cfb)

![image](https://github.com/user-attachments/assets/508aafc6-43b5-44f5-bcaf-c9746b2e2d76)

To end the exploratory part of the project we analyzed continuous variables by plotting our key variable 

![image](https://github.com/user-attachments/assets/f8c25ea5-92e6-499a-9d4c-9dffee5d8a15)

![image](https://github.com/user-attachments/assets/c628743b-38fe-4bda-8458-7b1c087a1e25)

Following that, we analyzed trends in fuel efficiency and model year over time.
![image](https://github.com/user-attachments/assets/87a6c81d-041e-4c1d-94d0-4732a38454dd)
![image](https://github.com/user-attachments/assets/345a570d-0345-4191-bc26-8867aebaa0ec)

#### Preprocessing 
##### For the preprocessing, first step was to drop unwanted columns and check for null values and duplicates. 
![image](https://github.com/user-attachments/assets/5b2e9cdf-f9ba-4bb8-a143-f72dd88d435e)

![image](https://github.com/user-attachments/assets/c80c450e-847a-44cb-9b91-811537eb6382)

##### After dropping null values for our target variable and replacing nulls in other columns with median, we changed the data type for columns 'displacement' and 'horsepower' as shown below:

![image](https://github.com/user-attachments/assets/7f074e3d-8ce3-4e01-9fdf-8dc84c088d0f)

For feature extraction, we splitted the 'car name' column into brand and model. Made some corrections for the name of the brand and one-hot encoded the final ones into the five largest brands and others.

![image](https://github.com/user-attachments/assets/ba253599-936a-48cc-916e-e79908824dce)

Next step on preprocessing was identifying and handling outliers. For identifying outliers for columns 'mpg', 'cylinders', 'displacement', 'horsepower', 'weight' we used the IQR Method as shown below and for column 'acceleration' which follows normal distribution we used Z-Score Method.

![image](https://github.com/user-attachments/assets/bef3aab0-63ca-4224-91ae-8a2324dd0c08)

![image](https://github.com/user-attachments/assets/7d012506-8b62-4d7c-95a5-a5565ce26855) 


Given that our dataset was relatively small and the outliers fell within a reasonable range, we decided to retain them.

The final preprocessing step involved splitting the mpg column into three classes based on percentiles. This allowed us to reframe the problem as a classification task, making it suitable for model training and evaluation. By doing this, we created a new efficiency column that classifies fuel efficiency into three categories: low (3), medium (2), and high (1) based on the mpg values.

![image](https://github.com/user-attachments/assets/83e3d484-f47e-40e5-94c3-6a179d6fd912)


### Model Training, Testing & Evaluation
* To start, we scaled the data to ensure no feature had a disproportionate influence on model performance due to its magnitude.
  
![image](https://github.com/user-attachments/assets/67849fa9-fac8-43c3-8ef2-28604a3355a0)


* Next, we splitted the dataset into training and testing sets. We applied four popular classification algorithms: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN) and XGBoost evaluating their performance with and without class weights to address the class imbalance.
  
Model Shown: Logistic Regression without class weights

![image](https://github.com/user-attachments/assets/c813211d-a01a-4341-b78c-74c7f234d3ac)

Model Shown: XGBoost without class weights

![image](https://github.com/user-attachments/assets/60e5d906-5152-4b13-aaf2-99025ba880e5)


* After training and evaluating the models, we found that the Logistic Regression and XGBoost performed best, based on F1-score and macro average across all classes.
  
* XGBoost model achieved an accuracy of 88.75%, performing well overall. It predicts high fuel efficiency (class 3) perfectly with a recall of 100%, meaning all actual class 3 cars were correctly classified. However, it struggles with low fuel efficiency (class 1), achieving only 62% recall, which means many low-efficiency cars were misclassified. The model performs best on medium and high-efficiency cars, with precision above 87%. To improve class 1 recall, we might consider adjusting class weights.
  
* Logistic Regression without class weights balanced precision and recall effectively, achieving an overall accuracy of 93%. Class 3 (High Efficiency) has perfect recall (1.00), meaning no instances were missed and class 1 (Low Efficiency) has the lowest recall (0.75), indicating that some instances were misclassified. The model performs well overall, but class imbalance may affect the recall of the minority class.

### Model Fine-Tuning

To further optimize the Logistic Regression model, we employed cross-validation and GridSearchCV techniques.

##### Cross-Validation Results:
* Mean Accuracy: 0.8175
  Standard Deviation: 0.0381

The mean accuracy of 0.8175 indicated that the Logistic Regression model was performing reasonably well, but there was room for improvement. The standard deviation of 0.0381 showed that the model is consistent and reliable across different data subsets.
 
![image](https://github.com/user-attachments/assets/4dce0f61-fe3e-474a-aaf7-d64a892ba51e)


#### For optimizing further the hyperparameters

#### GridSearchCV Results:

After applying GridSearchCV, we tuned the hyperparameters to enhance the model's performance.

![image](https://github.com/user-attachments/assets/7629a65f-407a-434c-84f9-7ea7921fcb32)

#### Comparison of Results

##### Before Hyperparameter Tuning:
* The Logistic Regression model performed well with a high recall for class 3 (1.00) and good precision and F1-score for all classes.
  Overall accuracy: 93%
  Macro avg F1-score: 0.89

##### After Hyperparameter Tuning (with GridSearchCV):
* GridSearchCV improved the recall for class 3 (0.94).
  However, accuracy decreased slightly to 85%, and the macro F1-score dropped to 0.81.
  The model became more balanced, with better performance in terms of recall but a trade-off in overall accuracy and class 1 performance (recall dropped to 0.38).
  This process highlights how GridSearchCV optimization can fine-tune a model, leading to a more balanced result but with some trade-offs in performance.

#### Summary of the project
This project focused on successfully reframing a regression problem into classification. We preprocessed data effectively for machine learning and tested multiple models while fine-tuned the best performer.Provided a robust and scalable solution for predicting MPG categories, with the help of KNN which helped us 
The result of this project is a classification model that categorizes cars based on their fuel efficiency into three classes: high, medium, and low efficiency. The model uses various machine learning techniques, including XGBoost, to predict these classes based on features such as miles per gallon (mpg) and other relevant car attributes.

### Key Outcomes

#### Model Performance: 
##### Logistic regression without class weights
Accuracy: The model achieved an overall accuracy of 93%, which indicates that the model's predictions are generally reliable.
*  Low-efficiency cars (Class 3) are perfectly captured.
*  High-efficiency cars (Class 1) need better recall (reduce misclassification).
*  Medium-efficiency cars (Class 2) still have a 10% misclassification rate.

#### Model Performance:
##### XGBoost without class weights
Accuracy: The model achieved an overall accuracy of 88.75%, indicating strong performance.
* High-efficiency cars (Class 3) are perfectly captured with 100% recall.
* Low-efficiency cars (Class 1) need better recall, as 38% were misclassified.
* Medium-efficiency cars (Class 2) perform well but still have a 15% misclassification rate.

#### Practical Application:

The fuel efficiency classification model provides valuable insights for businesses in the automotive and transportation sectors. Here’s how it can be applied:

* Fleet Optimization for Car Rentals: Car rental agencies can use the model to select cars with the best fuel efficiency, optimizing their fleet mix to meet customer demand while controlling costs. This helps businesses offer a range of vehicles while maintaining low fuel consumption.

* Environmental Sustainability: By selecting high-efficiency vehicles, businesses can reduce their carbon footprint and align with sustainability goals. This helps businesses stay environmentally responsible while enhancing their brand image among eco-conscious customers.

* Fuel Cost Optimization: Fuel efficiency directly impacts operating costs. By adopting the model, businesses can reduce fuel consumption, saving money on fuel expenses over time. This contributes to improved profitability and competitiveness.

* Consumer Insights:Companies can use the model to understand consumer preferences for fuel-efficient vehicles. This helps in marketing and targeting eco-conscious customers with the right options, improving customer satisfaction and loyalty. 

*  Data-Driven Strategy:The model provides a data-driven tool for fleet management and long-term planning. By considering fuel efficiency, businesses can adapt to changing market conditions, optimize costs, and stay ahead of industry trends.


####  The model empowers businesses to make smarter decisions in fleet management, reduce fuel costs, enhance sustainability, and improve customer satisfaction. It provides a competitive edge by optimizing fleet composition and supporting environmentally friendly business practices.




















