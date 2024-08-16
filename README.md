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







