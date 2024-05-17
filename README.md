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

### Model Training and Evaluation
Our first strategy was to implement a Demand Forecasting Model in order to predict the number of taxis needed at every hour based on historical data to optimize taxi dispatches. The preprocessing we did included time slots: Index time into 15, 30, and 60-minute intervals, features: Day, Time, Month, Pickup Location, Holiday, Time Slot, as well as grouping data by PickupLocationID and time slot. The model we implemented was a Random Forest, with the evaluation strategy of 80-20 split on the grouped dataset. The metric we employed was Root Mean Square Error (RMSE).

Our second model training implementation is a rider duration prediction model. The objective is predict trip duration using pick-up and drop-off data to uncover insights about traffic patterns. Our preprocessing included location data: Convert PULocationID and DOLocationID to latitude and longitude for precise location analysis. Features were Day, Time, Month, Latitude, Longitude. The model type is XGBoost Regressor, and evaluation strategy was once again, 80-20 split, with the metric RMSE. 

Our third model training implentation was focused on anomaly detection. The objective was to identify irregularities such as unusual fare prices which could indicate errors or fraudulent activity. Preprocessing included the features: Day, Time, Fare Price. The model was type PCA (Principal Component Analysis). Our task succeeded in applying PCA to reduce dimensionality and detect outliers based on fare price variations. 

Implementation included PySpark, since it is able to handle large datasets efficiently, PySpark MLlib for Random Forest and XGBoost models, and PCA for anomaly detection.
