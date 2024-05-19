# YellowTaxi

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