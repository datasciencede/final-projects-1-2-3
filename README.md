# FINAL- PROJECT- 1

1. CAR INSURANCE PREDICATION

 1.Imports and Setup:    
          
          The code imports various libraries needed for data manipulation, visualization, and machine learning. 
          It also sets up a consistent style for plots and suppresses warnings.

2. Data Loading and Caching:

            A function load_data loads the dataset from a specified file path and caches.
             It for faster access during the app's runtime.

3. Data Preprocessing:

                     The preprocess_data function handles cleaning and transforming the dataset:
                       1.Drops irrelevant columns.
                       2.Converts categorical variables to numerical values.
                       3.Cleans and converts monetary columns to floats.
                       4.Fills missing values with appropriate statistics (mean or median).

4. Visualization:

                  The app provides options to visualize the dataset:
                    1.plot_distribution function shows the distribution of the CLM_AMT (claim amount).
                    2.corrplt function generates a correlation heatmap of the dataset's numerical features.

5. Model Building:

                   1.Users can select features and a model type (Linear Regression, Decision Tree, or SVM) to train on the dataset.
                   2.The model is trained, evaluated, and its performance metrics (e.g., Mean Squared Error, Accuracy) are displayed.
                   3.For SVM, the problem is treated as a classification task (claim vs. no claim).

6. Prediction:

               Users can input values for selected features and the trained model will predict.
               The claim amount or classify whether a claim will be made.

7. Interactivity:

                The app's interface allows users to navigate between
                1.showing data, visualizing distributions, building models, and making predictions using a sidebar.

 ------------------------------------------------------------------*******************---------------------------------------------------

# FINAL- PROJECT- 2

2.FAKE NEWS CLASSIFICATION

  1. Imports and Setup

                    Libraries:

                               The code imports necessary libraries for
                                1.data handling (pandas)
                                2.visualization (matplotlib, seaborn),
                                3.text processing (TfidfVectorizer),
                                4.machine learning (LogisticRegression)
                                5.performance evaluation (accuracy_score, confusion_matrix, etc.)
      
                    File Paths:

                               The paths to the training, testing, and predictions datasets are defined.



  2. Data Loading:
 
                    Loads training and test datasets, filling missing values in key columns with "unknown."

     


  3. Feature Engineering:

                          Combines title, author, and text into a single feature, content, for modeling. 
   


  4. Menu Navigation:

                     Users can navigate between "Home", "EDA", "Model Training", "Prediction", and "Test Your Own Article" options


  5. Exploratory Data Analysis (EDA):

                                     Visualizes the distribution of the target variable, missing values
                                     article length, and generates word clouds for reliable and unreliable articles.


  6. Model Training:

                        Trains a logistic regression model using TF-IDF vectorized text.
                        Evaluates performance with accuracy, precision, recall, F1 score, and confusion matrix.   

7. Prediction:

                Generates and displays predictions on test data, with an option to save the results.



                         ------------------------------*******--------------------------
