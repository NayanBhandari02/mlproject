import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import dataTransformation,dataTransformationConfig
from src.components.model_trainer import modelTrainer,modelTrainerConfig

@dataclass
class dataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class dataIngestion:
    def __init__(self):
        self.ingestion_config=dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Component")
        try:
            df=pd.read_csv('data/stud-data.csv')  
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # create directory/folder for trained data csv file
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated!")

            x_train,x_test = train_test_split(df,test_size=0.2,random_state=12)

            x_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            x_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=dataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = dataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=modelTrainer()
    r2score, best_model = model_trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
    print('Best R2 Score: {:.2f}%'.format(r2score * 100))
    print('Model Name:{}'.format(best_model))