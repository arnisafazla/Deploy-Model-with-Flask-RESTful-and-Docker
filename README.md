# Train ML Pipelines using Docker and deploy the model locally using Flask-RESTFul
A complete ML pipeline run in a Docker container. The model is deployed locally as a REST API using Flask-RESTFul, and in production settings using gunicorn. 

The command line commands to run the pipeline (use sudo if running in a Unix environment):

1. Build the docker image - The model training will be done during the build step:

`docker build -t docker-api -f Dockerfile .`

2. Run the container and start online inference in development settings:

`docker run -it -p 5000:5000 docker-api python3 api.py`

3. Run the container and start the inference in production settings:

`docker run -it -p 5000:5000 docker-api gunicorn -b :5000 api:app --log-level=info`


This project was adapted from these two sources:

1. The Docker for Machine Learning blog series: <https://mlinproduction.com/docker-for-ml-part-1/>

2. MLOps with Docker and Jenkins: Automating Machine Learning Pipelines can be found in <https://towardsdatascience.com/mlops-with-docker-and-jenkins-automating-machine-learning-pipelines-a3a4026c4487>
