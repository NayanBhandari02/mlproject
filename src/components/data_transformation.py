import sys
from dataclasses import dataclass
import numpy as np
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from src.utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts',"preprocessor.pkl")

class dataTransformation:
    def __init__(self):
        self.data_transformation_config=dataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Responsible for data transformation
        '''
        try:
            numerical_cols = ["writing_score","reading_score"]
            categorical_cols=["ss","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            numerical_pipeline =Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ('Scaler',StandardScaler())
                ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy="most_frequent")),
                    ('one_hot-encoder',OneHotEncoder(sparse_output=False)),
                    ('Scaler',StandardScaler())
                ]
            )
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ('Numerical_pipeline',numerical_pipeline,numerical_cols),
                    ('Categorical_pipeline',categorical_pipeline,categorical_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_col_name="math_score"
            numerical_cols = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=target_col_name,axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=target_col_name,axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)