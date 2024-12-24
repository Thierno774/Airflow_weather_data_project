from airflow import DAG # type: ignore
import requests 
import pandas as pd 
import numpy as np 
from datetime import timedelta, datetime 
from airflow.providers.http.sensors.http import HttpSensor # type: ignore
from airflow.providers.http.operators.http import SimpleHttpOperator 
from airflow.operators.python import PythonOperator 
from airflow.providers.postgres.operators.postgres import PostgresOperator 
from airflow.hooks.postgres_hook import PostgresHook


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from joblib import dump


import json 
import os 

BASE_URL ="https://api.openweathermap.org/data/2.5/weather?q=" 
CITY_NAME = "Nantes"
API_KEY = "92a2228961d1701676a43daabb355f4d"

LIST_CITIES = ["Toulouse", "Nantes","Grenoble", "Marseille", 
               "Lyon", "Montreal", "New York", "Conakry", "Dakar"]

def extract_data(list_cities): 
    datas = {}
    for city in list_cities: 
        url = f"{BASE_URL}{city}&appid={API_KEY}"
        response = requests.get(url)
        datas[city] = response.json()
    ## Save data in directory 
    # The current time 
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    time_now = str(time_now)
    time_now = time_now+ '.json'
    #filename = f'/app/raw_files/test.json'


    current_dag_directory = os.path.dirname(os.path.abspath(__file__))
    #print(current_dag_directory)
    filename = os.path.join("/app/raw_files", time_now)

    ## Save json file 
    with open(filename, "w") as file: 
        json.dump(datas, file, indent=4)


def transform_data_intro_csv_top_20_files(**kwargs): 
    n_files = kwargs["n_files"]
    filename = kwargs["filename"]
    parent_folder = "/app/raw_files"
    files = sorted(os.listdir(parent_folder), reverse= True)
    dfs = []
    if n_files: 
        files = files[:n_files]
    for f in files:
        with open(os.path.join(parent_folder,f), "r") as file: 
            data  = json.load(file)
            keys = list(data.keys())
            for element in range(len(keys)):
                city = data[keys[element]]["name"] 
                city = city.replace("Arrondissement de", "")
                dfs.append ( {
                        "temperature" : data[keys[element]]["main"]["temp"], 
                        "city" : city,
                        "pression" : data[keys[element]]['main']["pressure"], 
                        'date': f.split('.')[0]
                        })
    dataset = pd.DataFrame(dfs)
    print(dataset.head())
    ## Directory for to save data
    directory_to_save_data = ("/app/clean_data")
    if not os.path.isdir(directory_to_save_data): 
         os.makedirs(directory_to_save_data)
    dataset.to_csv(os.path.join(directory_to_save_data, filename), index=False)




def transform_data_intro_csv_all_files(**kwargs): 
    filename = kwargs["filename"]
    parent_folder = "/app/raw_files"
    files = sorted(os.listdir(parent_folder), reverse= True)
    print(files)
    dfs = []
    for f in files:
        try:
            with open(os.path.join(parent_folder,f), "r") as file: 
                data  = json.load(file)
                keys = list(data.keys())
                for element in range(len(keys)):
                    city = data[keys[element]]["name"] 
                    city = city.replace("Arrondissement de", "")
                    dfs.append ( {
                            "temperature" : data[keys[element]]["main"]["temp"], 
                            "city" : city,
                            "pression" : data[keys[element]]['main']["pressure"], 
                            'date': f.split('.')[0]
                            })
        except: 
            pass
    dataset = pd.DataFrame(dfs)
    ## Directory for to save data
    directory_to_save_data = ("/app/clean_data")
    if not os.path.isdir(directory_to_save_data): 
         os.makedirs(directory_to_save_data)
    dataset.to_csv(os.path.join(directory_to_save_data, filename), index=False)




def prepare_data(path_to_data='/app/clean_data/fulldata.csv'): 
     # Reading data 
    data = pd.read_csv(path_to_data)
    datas = []
    # Ordering data according to the city 
    data = data.sort_values(["city", "date"], ascending=True)
    for c in data['city'].unique(): 
        data_temp = data[data['city'] ==c]
        # Creating target 
        data_temp.loc[:, "target"] = data_temp['temperature'].shift(1)
        # Creating the features 
        for element in range(1,10): 
            data_temp.loc[:, f"temp_m-{element}"] = data_temp['temperature'].shift(-element)
        #Deleting null values *
        data_temp = data_temp.dropna()
        datas.append(data_temp)
    # concatenating datasets
    data_final = pd.concat(datas, axis=0, ignore_index=False)
    ## Deleting data variables 
    data_final = data_final.drop(["date"], axis = 1)
    # Creating dummies for city variable 
    data_final = pd.get_dummies(data_final)
    # features = data_final.drop(['target'], axis = 1)
    # target = data_final["target"]
    ## return the result 
    return data_final


def Linear_Regression_Model(task_instance): 
    data = task_instance.xcom_pull(task_ids = "preprocessing_data")

    #features = data.drop(['target'], axis = 1)
    target = data["target"]
    feats =data.drop(["target"], axis = 1)
    

    # Linear Regression Model instance 
    model = LinearRegression()
    # Train model 
    model.fit(feats, target) 
     # Compunte cross val 
    cross_validation = cross_val_score(
                        LinearRegression(), 
                        feats, target, cv= 3, scoring="neg_mean_squared_error"
    )
    model_score = cross_validation.mean()
    print("The Negative Score with Linear Regression ", round(model_score, 3))

    return model_score


def Random_Forest_Model(task_instance): 
    data = task_instance.xcom_pull(task_ids = "preprocessing_data")

    #features = data.drop(['target'], axis = 1)
    target = data["target"]
    feats =data.drop(["target"], axis = 1)
    
    model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    # Train model 
    model.fit(feats, target) 
     # Compunte cross val 
    cross_validation = cross_val_score(
                        RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True), 
                        feats, target, cv= 3, scoring="neg_mean_squared_error"
    )
    model_score = cross_validation.mean()
    print("The Negative Score with Random Forest ", round(model_score, 3))

    return model_score


def Decision_tree_regressor(task_instance): 
    data = task_instance.xcom_pull(task_ids = "preprocessing_data")

    #features = data.drop(['target'], axis = 1)
    target = data["target"]
    feats =data.drop(["target"], axis = 1)
    
    model = regressor = DecisionTreeRegressor(random_state = 0)  

    # Train model 
    model.fit(feats, target) 
     # Compunte cross val 
    cross_validation = cross_val_score(
                        DecisionTreeRegressor(random_state = 0), 
                        feats, target, cv= 3, scoring="neg_mean_squared_error"
    )
    model_score = cross_validation.mean()
    print("The Negative Score with Decision Regressor ", round(model_score, 3))

    return model_score


def train_and_save_model(model, X, y, name_of_model):

    ## Directory for to save data

    path_to_model = ("/app/clean_data")
    # if not os.path.isdir(path_to_model): 
    #     os.makedirs(path_to_model)
    file = f'{name_of_model}_best_model.pickle'
    filename = os.path.join(path_to_model, file)

    # training the model
    model.fit(X, y)
    # saving model

    print(str(model), 'saved at ', filename)
    dump(model, filename)


def best_model_function(task_instance): 
    data, score_lr, score_random_forest, score_decision_tree = task_instance.xcom_pull(task_ids = ["preprocessing_data", "Linear_regression", "Random_Forest", "Decision_tree_regressor"])
    #features = data.drop(['target'], axis = 1)
    target = data["target"]
    feats =data.drop(["target"], axis = 1)
    if score_lr < score_random_forest and score_random_forest < score_decision_tree: 
        print (f"The best Model is Linear Regressor,  {score_lr} ")
        train_and_save_model(LinearRegression(), feats, target, "linear_regression")

    elif score_random_forest < score_lr and score_random_forest < score_decision_tree: 
        print (f"The best Model is  Random Forest, {score_random_forest} ")
        train_and_save_model( RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True), feats, target, "Random_forest")
    else: 
        print (f"The best Model is Decision Tree Regressor,  {score_decision_tree} ")
        train_and_save_model (DecisionTreeRegressor(random_state = 0), feats, target, "decision_tree_regresor")
    pass

    




default_args = {
            "owner": "airflow", 
            "depends_on_past" :False, 
            "start_date" : datetime(2024,12,6),
            "email" :["thiernosidybah232@gmail.com"], 
            "email_on_failure" : False, 
            "email_on_retry" : False, 
            "retries": 2, 
            "retry_delay" :timedelta(minutes=1)
}

# Create the DAG

with DAG( "weather_dag", 
         default_args = default_args, 
         schedule_interval = "* * * * *", 
         catchup = False) as dag : 

        extract_weather_data = PythonOperator(
                        task_id = 'Extract_weather_data', 
                        python_callable = extract_data, 
                        op_args = [LIST_CITIES]
        )
        transform_data_top20_files_task = PythonOperator(
                        task_id = "transform_data_top_20_files", 
                        python_callable = transform_data_intro_csv_top_20_files, 
                        op_kwargs={"n_files" : 20, 'filename': "data.csv"})
        
        transform_data__all_files_task = PythonOperator(
                                task_id = "transform_data_all_files", 
                                python_callable = transform_data_intro_csv_all_files, 
                                op_kwargs={'filename': "fulldata.csv"})
                
        
        prepare_data_task = PythonOperator(
                        task_id = "preprocessing_data",
                        python_callable = prepare_data)
        
        Linear_Regression_task = PythonOperator(
                                    task_id = "Linear_regression", 
                                    python_callable = Linear_Regression_Model
        )

        
        Random_forest_task = PythonOperator(
                                        task_id = "Random_Forest",
                                        python_callable = Random_Forest_Model
        )

        Decision_Tree_regressor_task = PythonOperator(
                                        task_id = "Decision_tree_regressor", 
                                        python_callable = Decision_tree_regressor
        )
        best_model_task = PythonOperator(
                                    task_id = "Best_model", 
                                    python_callable = best_model_function)


    
        extract_weather_data >> transform_data_top20_files_task 
        extract_weather_data >> transform_data__all_files_task >> prepare_data_task >>Linear_Regression_task
        prepare_data_task >> Random_forest_task
        prepare_data_task >> Decision_Tree_regressor_task
        Linear_Regression_task >> best_model_task
        Random_forest_task >> best_model_task
        Decision_Tree_regressor_task >> best_model_task