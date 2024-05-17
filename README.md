# YellowTaxi

## EDA

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

### Anomaly Detection
Through PCA technique, we aim to reduce dimensionality of the dataset, thereby enhancing our ability to identify outliers effectively.
Anomaly Criteria: Establish anomalies based on deviations from the norm where the PCA-transformed features exceed the mean plus three times the standard deviation of the  features.

### Model Training and Evaluation
Training Process- Model Setup: We will initially deploy a Random Forest model due to its robustness against overfitting and its ability to handle large datasets with a mixture of categorical and numerical data effectively.
Feature Engineering: To better capture the dynamics of taxi demand, we will experiment with polynomial feature expansion and interactions between key features such as time slots and locations.
