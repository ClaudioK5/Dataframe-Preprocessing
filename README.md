This project is focused on dataframe preprocessing, an essential step in the machine learning process, as approximately 80% of the work in machine learning involves preprocessing of raw data.

The project offers two primary methods:

- **Column Deletion**: This method removes columns with a percentage of null values exceeding a user-defined threshold.
- **Linear Regression**: This method uses linear regression to estimate and fill in missing values in columns with a manageable percentage of null values, preserving therefore the underlying pattern in the data. Users can select which columns to use as predictors for the model, as well as designate a target column where the predictions should be set, replacing null values. **Note: only columns containing numerical values with null values can be chosen.**

Both methods are customizable, allowing users to set parameters based on their specific needs.
The project uses only **sklearn** and **panda** as the external packages.
