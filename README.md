# Airflow_weather_data_project

# Extraction, Transformation, Analyse des données et Entrainement des Modèles de Machine Learning  et Ochestration avec Airflow
This project uses Apache Airflow to extract weather data from an API, transform the data, and load it in 
postgreSQL database. The workflow is designed to run daily and can be easily customized to extract weather data for any city

![Sparkify Data Model](/images/dag_weather_data.png)    



# Requirements for the project
* Python 3.6 or higher
* Apache Airflow
* Pandas
* Postgres
* pgAdmin
# Installation
Clone the repository:


```git clone https://github.com/Thierno774/ETL_Pipeline_with_Airflow```


# Set up Airflow:
You'll need to set up Airflow to run the workflow. Follow the instructions in the [official Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html) to install and configure Airflow.

# Add the DAG to Airflow:
Copy the main.py file to the dags folder in your Airflow installation directory. You may need to restart the Airflow scheduler and webserver for the DAG to appear in the Airflow UI.

# Set up the weather API connection:
In the Airflow UI, go to the Admin tab and select Connections. Click the Create button and fill in the following details:

* Conn Id: weathermap_api
* Conn Type: HTTP
* Host: api.openweathermap.org
* Port: 8080
* Login: (leave blank)
* Password: (leave blank)
* Extra: {"endpoint": "/data/2.5/weather?q=Nantes&APPID=YOUR_API_KEY"}
* Replace YOUR_API_KEY with your OpenWeatherMap API key.

# Set up the docker-compose:

-Pull the images:docker pull dpage/pgadmin4
- Add the images in docker compose
- EXpose the port: "5050:80"
- In the Browser, open the pgadmin, create a new database and connect the postgre by using the IP adress (docker inspect docker_name)

# Usage

Once you've set up the DAG and connections, the workflow will run automatically on the schedule you've set (default is daily). You can monitor the progress of the workflow in the Airflow UI.

The transformed weather data will be PostgreSQL database.

# Machine Learning

To extract weather data for a different city, modify the endpoint parameter in the HttpSensor and SimpleHttpOperator tasks in the weather_dag.py file. You'll also need to update the transform_load_data function to extract the relevant data for the new city.

# Dashboard 

It is often useful to visualize complex data flows using a graph. Visually, a node in a graph represents a task, and an arrow represents the dependency of one task on another. Given that data only needs to be computed once on a given task and the computation then carries forward, the graph is directed and acyclic. This is why Airflow jobs are commonly referred to as “DAGs” (Directed Acyclic Graphs).  ![Sparkify Data Model](/images/dash_weather_data.png)    

# See The Data table in database

In the  pgadmin, see the result by using the SQL request (SELECT * FROM Weather_table; ![Sparkify Data Model](/images/postgres_images.png)    

