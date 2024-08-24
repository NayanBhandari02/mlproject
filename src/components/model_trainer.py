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
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

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
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbor Regressor": KNeighborsRegressor(),
                #"Cat Boosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['gini','squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbor Regressor":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [32,64,128,256]
                },
                #"CatBoosting Regressor":{
                #    'depth': [6,8,10],
                #    'learning_rate': [0.01, 0.05, 0.1],
                #    'iterations': [30, 50, 100]
                #},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }   
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)

            logging.info("Model evaluation scores: {}".format(model_report))

            #best_model_score = max(sorted(model_report.values()))

            #best_model_name = list(model_report.values())[list(model_report.values()).index(best_model_score)]
            #best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model_name = max(model_report, key=lambda k: model_report[k]['test_r2_score'])
            best_model_r2score = model_report[best_model_name]['test_r2_score']
            best_model_mse = model_report[best_model_name]['test_mse']
            best_model_mae = model_report[best_model_name]['test_mae']
            
            best_model = models[best_model_name]

            if best_model_r2score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found:{}".format(best_model))
            logging.info("Best model R² score: {}".format(best_model_r2score))
            logging.info("Best model MSE: {}".format(best_model_mse))
            logging.info("Best model MAE: {}".format(best_model_mae))

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted = best_model.predict(x_test)
            r2score = r2_score(y_test,predicted)
            mse = mean_squared_error(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            return r2score, mse, mae, best_model, model_report
        
        except Exception as e:
            raise CustomException(e,sys)