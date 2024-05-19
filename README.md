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
