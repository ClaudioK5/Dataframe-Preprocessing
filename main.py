import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import argparse

class DataFramePreprocessing:

    """
    
    The DataFramePreprocessing class provides essential tools for preparing data
    before using it in machine learning models. Data preprocessing is a crucial step, often 
    accounting for around 80% of the work required to make data suitable for machine learning 
    algorithms.
    
    """

    def __init__(self, dataframe_path, percentage):

        self.dataframe = pd.read_csv(dataframe_path)
        self.percentage = percentage
        

    def cleaner(self):

        """ 
        this method gets rid of those columns that have a percentage of null values that
        exceeds the user-defined threshold (percentage parameter).
        """
        
        df_titles = self.df.columns.tolist()
        
        for x in df_titles:
            
            if self.df[x].count()/(df[x].count()+self.df[x].isnull().sum())<self.percentage:
                
                self.df.drop(x,axis=1,inplace=True)

            else:
            
                pass

    def LinearRegression(self, a, b):

        """
        
        This method takes two parameters:
        - 'a' : the names of the columns used to train the linear regression model.
        - 'b' : the name of the column where null values are present and need to be replaced.

        It uses linear regression to predict and fill in missing values in column 'b'.
        
        """

        data = self.dataframe.dropna(subset=[b])

        X = data[a]

        y = data[b]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

        lm = LinearRegression()
    
        lm.fit(X_train, y_train)
        
        null_rows = self.dataframe[self.dataframe[b].isnull()]

        null_rows = null_rows[a]

        null_rows.head(10)

        predictions = lm.predict(null_rows[a])

        df.loc[null_rows.index, b] = predictions


def parse_arguments():

    parser = argparse.ArgumentParser(description = "DataFramePreprocessing")

    parser.add_argument("dataframe", type = str)

    parser.add_argument("percentage", type = float)

    parser.add_argument("a", type = str, nargs='+')

    parser.add_argument("b", type = str)


    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_arguments()

    dataframePreprocesser = DataFramePreprocessing(args.dataframe, args.percentage)

    dataframePreprocesser.cleaner()

    dataFramePreprocesser.LinearRegression(args.a, args.b)
