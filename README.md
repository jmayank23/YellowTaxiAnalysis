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

For our second model, we similarly chose an XGBoost regressor, but for a different task. We took the latitude and longitudes of pickup and dropoff locations, day, time, and month to predict the duration of a given trip. Similarly to our first model, we opted for a 80/20 train-test split with an evaluation metric of RMSE. The hyper paramteters for the model were the default for SparkXGBRegressor.

#### Model 2

With our features preprocessed using the methods described above, we fed the data into an XGBoost regressor with and 80/20 train-test split of 6 months of data. The goal of this model was to take the features of day, time, month, pickup location, holiday, and timeslot to predict the number of taxis needed at every hour timeslot to meet demand. The hyper parameters for the model were the default for SparkXGBRegressor.

### Result

#### Model 1

- Data Pre-Processing and Feature Engineering:
Data preprocessing involved merging trip data with geographical zone information, converting timestamps into more granular time features (e.g., month, day, hour), and encoding categorical data. Missing values were imputed based on their mean or mode as appropriate, ensuring robustness in the model training process.

- Model Evaluation:
XGBoost and Linear Regression models were trained on an 80/20 split of the data. The Linear Regression model utilized standardized features to prevent scale discrepancies from influencing the model’s performance. The XGBoost model was configured with default settings, focusing on capturing nonlinear patterns and interactions between features. The performance of the models was assessed using the Root Mean Squared Error (RMSE) metric, computed on the test dataset. The Linear Regression model achieved an RMSE of 42.43, indicating the average prediction error in minutes. The XGBoost model demonstrated a slightly better performance with an RMSE of 41.25.
#### Model 2

- Data Pre-Processing:
We extracted day, month, and hour from the tpep_pickup_datetime to enrich our dataset with additional temporal features. Utilizing the USFederalHolidayCalendar, we identified public holidays to examine their impact on taxi usage. We created indices representing 15-minute, 30-minute, and 60-minute intervals throughout the day to analyze patterns in taxi demand.

- Model Evaluation:
We applied two predictive models to forecast taxi demand based on the processed features: XGBoost and Linear Regression. Both models were trained on an 80/20 split of the data. We configured an XGBoost regressor with the aim to predict the number of taxi trips in 60-minute intervals. The RMSE (Root Mean Squared Error) achieved by the XGBoost model was 96.66, indicating a relatively high accuracy in capturing the variability of the taxi demand. We also employed a Linear Regression model for the same prediction task. However, this model resulted in a significantly higher RMSE of 2817.12, suggesting that it was less effective at modeling the complex patterns present in our dataset.

### Discussion

#### Model 1

- Model Selection and Rationale:
The choice of XGBoost and Linear Regression models was driven by their contrasting capabilities and the nature of our dataset. XGBoost, known for its performance in complex datasets with non-linear relationships, was expected to handle the intricacies of taxi trip data effectively. Linear Regression, while simpler and less powerful for non-linear patterns, was chosen to serve as a baseline for comparison. This methodological diversity allowed us to evaluate the spectrum of model capabilities and their suitability for predicting taxi demand.

- Interpretation of Results:
The performance difference between XGBoost and Linear Regression was significant. The XGBoost model's RMSE of 96.66 suggests a strong fit to the data, indicating its robustness in capturing the complex dynamics of taxi usage patterns, including temporal variations and holiday impacts. In contrast, the Linear Regression model's high RMSE of 2817.12 highlights its limitations in this context, primarily due to its inability to model the non-linear dependencies between features and taxi demand effectively.

- Shortcomings and Critique:
Our analysis was constrained to one year of data and focused on aggregate demand without distinguishing between different types of taxi services or varying geographic details beyond pickup locations. Including additional data, such as weather conditions or special events, could potentially enhance model accuracy and relevance. Moreover, while we incorporated time and holiday features, our models might benefit from more sophisticated features such as interaction terms between time slots and location IDs, or more granular temporal resolutions for holiday effects.

- Potential Improvements:
1. Incorporating More Data: Adding data from multiple years or different types of taxis could provide a more comprehensive understanding of patterns and improve the robustness of predictions.
2. Advanced Feature Engineering: Exploring more complex features and transformation techniques could unveil deeper insights and enhance model performance.

#### Model 2

- Model Selection and Rationale:
For predicting taxi trip durations, we utilized both Linear Regression and XGBoost Regression models to cover a range of modeling techniques from simple to more complex approaches. The Linear Regression model served as a straightforward approach to gauge basic relationships between features and the target variable. On the other hand, XGBoost was chosen for its ability to handle complex patterns and interactions among features, which are common in geospatial and temporal data like taxi trip records.

- Interpretation of Results:
The results indicated close performance between the two models, with XGBoost slightly outperforming Linear Regression (RMSE of 41.25 vs. 42.43). This suggests that while the trip duration can be reasonably predicted by linear relationships, there are subtle non-linear interactions captured by XGBoost that contribute to a more accurate model. The relatively small difference in RMSE, however, points to the possibility that the most impactful predictors (like trip distance and time of day) may have linear relationships with trip duration.

- Shortcomings and Critique:
The minimal performance gain achieved by XGBoost compared to the simpler Linear Regression model raises questions about the cost-benefit of deploying more complex models. Given the additional computational cost and complexity in tuning and interpretation associated with XGBoost, the slight improvement might not justify its use in all application scenarios. Our approach to feature engineering was primarily basic temporal and spatial transformations. More sophisticated methods, such as interaction terms or polynomial features, might reveal deeper insights. Additionally, the imputation of missing values using mean or mode is a rudimentary approach that could be enhanced by more predictive imputation methods, potentially affecting model accuracy.

- Potential Improvements:
1. Enhanced Feature Engineering: Implementing more advanced feature engineering techniques, such as clustering geospatial data or engineering route complexity features from pickup and dropoff coordinates, could potentially unveil complex patterns not captured by the current model.
2. Cross-Validation: Employing more rigorous validation techniques, such as K-fold cross-validation, would ensure that the model's performance is consistent across different subsets of data and not just a result of particular train-test split.
3. Incorporation of Additional Data: Including external factors like weather conditions, traffic data, and special events could significantly enhance the model’s ability to predict trip durations more accurately.







