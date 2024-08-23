import os
import sys
from dataclasses import dataclass
#from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model
from sklearn.metrics import r2_score

@dataclass
class modelTrainerConfig():
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])

            models={
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbor Classifier": KNeighborsRegressor(),
                #"Cat Boosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "XGBClassifier": XGBRegressor()
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            #best_model_name = list(model_report.values())[list(model_report.values()).index(best_model_score)]
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found:{}".format(best_model))

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted = best_model.predict(x_test)
            r2score = r2_score(y_test,predicted)
            return r2score, best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)