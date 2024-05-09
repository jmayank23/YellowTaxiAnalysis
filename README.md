# YellowTaxi

For our EDA shown in the notebook we did a large amount of preprocessing. We chose to work with 6 months of data, but sampled 10% of this to ensure our code ran efficiently and plots were not overcrowded.

When visualizing the raw data for all of our EDA tasks, we initially noticed that there were many many outlier data points that made interpreting the data more difficult. An easy way to remediate this was filtering the data where feature values fall between a lower and upper bound determined by the 10th and 90th percentiles. For each EDA task, we constructed new filtered dataframes which led to more insightful EDA visualizations. For some of the features, we tried to capture the outlier data and analyze it as well.
