# YellowTaxi

## Introduction

Our project delves into the dense dataset of yellow cab taxi records provided by the New York City Limousine and Taxi Commission. This extensive dataset encompasses the last five years and includes critical features like pick-up and drop-off times, location IDs, trip distances, fares, rate types, payment methods, and passenger counts. One of our main motivations behind selecting this project stems from our fascination with urban mobility and its impact on city life. By leveraging machine learning, we aspired to predict taxi demand based on historical data, which could revolutionize how taxis are distributed across the city. Such predictive capabilities would not only enhance operational efficiency but also could ensure better service availability for New Yorkers.

The project is particularly exciting because it merges data analytics with real-world applications, offering a window into urban dynamics without direct traffic data. Through our models, particularly the XGRegression used for both demand forecasting and rider duration prediction, we hope to uncover hidden patterns and dependencies in urban travel. This includes understanding factors that influence taxi demand and analyzing ride durations to infer traffic patterns.

The broader impact of a well-calibrated predictive model in this context is substantial. For city planners and taxi companies, it means optimizing resources, reducing wait times, and improving customer satisfaction. For the environment, better vehicle distribution translates into reduced idle times and lower emissions. On a societal level, efficient taxi services can enhance the accessibility of urban areas, contributing to economic activities and overall urban livability. In essence, this project not only serves as an academic exercise but also as a pivotal tool for smarter, more sustainable urban planning.

## Milestone 1: Abstract

This study explores a dataset of yellow cab taxi records from New York City (provided by NYC Limousine and Taxi Commission). The dataset includes features such as pick-up and drop-off times, location IDs, trip distances, fares, rate types, payment methods, and passenger counts from the last 5 years. This study aims to conduct two machine learning tasks: develop demand forecasting models to predict taxi needs based on historical data, thereby enabling optimized vehicle distribution and revealing factors that influence taxi demand, and create a prediction model for rider duration using pick-up and drop-off to potentially uncover insights about traffic patterns, despite the absence of direct traffic data. Additionally, through exploratory data analysis, this study intends to perform anomaly detection to identify irregularities such as abnormal wait times and route deviations.

## Milestone 2: EDA

The results from our EDA are shown in the notebook. We decided to work with 6 months of data and sampled 10% of this to ensure our code ran efficiently and plots were not overcrowded.

When visualizing the raw data for all of our EDA tasks, we initially noticed that there were many many outlier data points that made interpreting the data difficult. To filter out this noise, we filter the data such that the feature values fall between a lower and upper bound, we determined that data between the 10th and 90th percentiles provides a good signal to understand the data features. For some of the features, we tried to capture the outlier data (for example, negative tip and fare amounts) and analyze it to find any trends or understand if the data might have been corrupted (if the negative amount was always -1000, say, then it is likely this would have happened from data corruption. However, we did not see such a constant value and plan to analyse further).

### Feature Relationships Analyzed in EDA

- Total Amount vs. Fare Amount vs. Tip Amount
- Trip Duration vs. Distance
- All Features (Trends by Hour of Day)
- Payment Type vs Locations
- Dropoff Location (Density)

## Preprocessing Plan

Going forward, we plan to use standard preprocessing strategies like converting our categorical variables to one-hot encoded vectors and for normalization of all numerical features we plan to subtract the mean and scale to unit variance, using Standard Scaler (from pyspark.ml.feature).

### Handling outliers and missing data

Our strategy to remove outliers by filtering data between the 10th and 90th percentile worked well on the sampled data, so we plan to implement the same strategy to handle outliers across the entire dataset that we will use for machine learning training tasks. Additionally, for any missing data we plan to explore either imputation using the median value for the feature or try to make it more granular and logical by instead getting the median value for a particular station (say) where we encounter the missing data point.

## Milestone 3: Finish Preprocessing and Train ML model

### Anomaly detection to identify irregularities

#### Preprocessing and Feature Selection

- Rows with any missing values were removed as it may be falsely detected as abnormal data.
- Trip Duration was calculated seconds by subtracting the pickup time from the dropoff time and added as a new colume ("trip duration")
- Trips with non-positive durations were filtered out from the analysis.
- Selected Feature: "trip_duration", "trip_distance", "fare_amount", "total_amount", "tip_amount", "tolls_amount", "congestion_surcharge", "airport_fee"
- VectorAssembler was used to combine the selected features into a single vector column named features.

#### PCA/Spectral Analysis

- Principal Component Analysis (PCA) was applied to reduce the dimensionality of the features column.
- UDF was defined and registered to calculate the Euclidean norm of PCA features.
- Summary statistics (mean and standard deviation) for the pca_norm column were calculated.
- Anomaly threshold was defined as the mean plus three times the standard deviation of pca_norm.
- Based on the defined threshold, anomalies were identified.

### Train Linear Regression Model and XGBoost Regressor to predict trip duration
- Given the features of day, time, month, pickup and dropoff locations, predict the ride duration.
- We represent the pickup and dropoff locations as latitude and longitude rather than Zone ID's.
To obtain the latitude and longitude of the locations, we used GeoPandas to read and process the shapefile containing the geometry of the location IDs. We then merged this shapefile with our Spark DataFrame. We found the representative points (centers) of the geometries and converted the coordinate system to obtain the latitude and longitude in degrees.
- Normalized all features using a standard scalar with a mean value of 0 and standard deviation of -1.
- We trained a Linear and XGBoost Regression model on these features and evaluate the predictions using RMSE.


From this task, we saw a training error of 0 with a test RMSE of ~41. Thus, our model would be on the far right of the fitting graph indicating a very high complexity with bad generalization on the data. For next time, we plan to use the same model, but rather improve our feature engineering, as we suspect this as the source of our prediciton results. Overall, we conclude our model overfitted the training set and we could improve it by changing how we are representing our features.

## Milestone 4: Final Submission

### Method

#### Data Exploration
We decided to work with 6 months of data and sampled 10% of this to ensure our code ran efficiently and plots were not overcrowded.
When visualizing the raw data for all of our EDA tasks, we initially noticed that there were many many outlier data points that made interpreting the data difficult. To filter out this noise, we filtered the data such that the feature values fall between a lower and upper bound. 

```
def filter_quantile_range(df, col_names, lower_quantile=0.01, upper_quantile=0.99):
    """
    Filter data in the DataFrame based on the specified quantile range for given columns.

    Parameters:
        df (DataFrame): The input DataFrame to filter.
        col_names (list): List of column names to filter on.
        lower_quantile (float): The lower quantile bound (e.g., 0.1 for the 10th percentile).
        upper_quantile (float): The upper quantile bound (e.g., 0.9 for the 90th percentile).

    Returns:
        DataFrame: The filtered DataFrame within the specified quantile range.
    """
    # Initialize an empty list to store the filter conditions
    conditions = []
    
    for column_name in col_names:
        # Get the quantile bounds for the column
        lower_bound, upper_bound = df.approxQuantile(column_name, [lower_quantile, upper_quantile], 0.01)
        
        # Create a condition to filter data within the quantile range for this column
        condition = (col(column_name) >= lower_bound) & (col(column_name) <= upper_bound)
        conditions.append(condition)
    
    # Combine all the conditions using AND (every column must meet its own condition)
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition &= condition
    
    # Filter the DataFrame based on the combined condition
    filtered_df = df.filter(combined_condition)
    
    return filtered_df

```
From this step determined that data between the 10th and 90th percentiles provides a good signal to understand the data features. 
```
filtered_df = filter_quantile_range(sampled_df, ["total_amount", "tip_amount", "fare_amount"], lower_quantile=0.0
filtered_pandas_df = filtered_df.select("total_amount", "tip_amount", "fare_amount").toPandas()
filtered_pandas_df.describe()
 ```
For some of the features, we tried to capture the outlier data (for example, negative tip and fare amounts) and looked for any trends or if the data might have been corrupted. If the negative amount was always -1000, say, then it is likely this would have happened from data corruption. However, we did not see such a constant value and plan to analyze further).
```
# Filter for negative fares and tips, group by payment type
negative_fares_tips = sampled_df.filter((col("fare_amount") < 0) | (col("tip_amount") < 0))
negative_fares_tips.groupBy("payment_type").count().show()
# Group by RateCodeID for entries with negative fares or tips
negative_fares_tips.groupBy("RateCodeID").count().show()

# Extract year and month, then analyze counts of negative values
negative_fares_tips.withColumn("year", year("tpep_pickup_datetime")) \
    .withColumn("month", month("tpep_pickup_datetime")) \
    .groupBy("year", "month").count().orderBy("year", "month").show()
```
Feature Relationships Analyzed in EDA
Total Amount vs. Fare Amount vs. Tip Amount
Trip Duration vs. Distance
All Features (Trends by Hour of Day)
Payment Type vs Locations
Dropoff Location (Density)

#### Preprocessing
In this project, we approached preprocessing with the purpose of enhancing our models. We removed rows with any missing values to prevent false anomaly detection and calculated the 'trip_duration' in seconds by subtracting the pickup time from the dropoff time, excluding non-positive durations. 
```
# Filter for negative fares and tips, group by payment type
negative_fares_tips = sampled_df.filter((col("fare_amount") < 0) | (col("tip_amount") < 0)) 
# Calculate the time difference in seconds using unix_timestamp
sampled_df = sampled_df.withColumn(
    "trip_duration_seconds",
    unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))
)

# filter using quantile range function based on time_difference_seconds
filtered_df = filter_quantile_range(sampled_df, ["trip_duration_seconds", "trip_distance"], 0.05, 0.95)

 ```
Key features such as 'trip_distance', 'fare_amount', 'total_amount', 'tip_amount', 'tolls_amount', 'congestion_surcharge', and 'airport_fee' were combined into a single vector using VectorAssembler.
```
# Select relevant features for analysis
features_df = df_cleaned.select("trip_duration", "trip_distance", "fare_amount", "total_amount", "tip_amount", "tolls_amount", "congestion_surcharge", "airport_fee")

# Assemble features into a vector
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=features_df.columns, outputCol="features")
vector_df = assembler.transform(features_df)

```
For dimensionality reduction, we applied Principal Component Analysis (PCA) and used a user-defined function (UDF) to compute the Euclidean norm of PCA features. By establishing an anomaly threshold based on statistical analysis, we were able to identify outliers effectively.
```
# Apply PCA
pca = PCA(k=3, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(vector_df)
pca_result = pca_model.transform(vector_df)

# Show PCA results
pca_result.select("pca_features").show(truncate=False)

 # Define a UDF to compute the norm (Euclidean distance) of the PCA features
def pca_norm(pca_vector):
    return float(Vectors.norm(pca_vector, 2))

norm_udf = udf(pca_norm, DoubleType())
# Add a column for the norm of the PCA features
pca_result = pca_result.withColumn("pca_norm", norm_udf(col("pca_features")))

 ```
In predicting trip duration, we shifted from using zone IDs to latitude and longitude for pickup and dropoff locations, utilizing GeoPandas for geographic data processing. 
```
# Apply PCA
pca = PCA(k=3, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(vector_df)
pca_result = pca_model.transform(vector_df)

# Show PCA results
pca_result.select("pca_features").show(truncate=False)
```
Features were normalized using a standard scalar, and we employed both Linear Regression and XGBoost models for prediction. 

#### Model 1
Despite achieving a training error of zero, the test Root Mean Squared Error (RMSE) was approximately 41, indicating overfitting. This result suggests our models were too complex, failing to generalize well to unseen data. Moving forward, we plan to refine our feature engineering approach to improve model performance and avoid overfitting, potentially leading to more robust and generalizable predictions.

#### Model 2
