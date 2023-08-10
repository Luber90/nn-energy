The compose folder contains docker compose configuration from https://github.com/hubblo-org/scaphandre with modified grafana dashboard for visualisation of this project data.

# Requiriments
To run the project using docker, nvidia-docker toolkit is required.
The dataset (COCO 2017 Unlabeled was used) should be located in the directory named unlabeled2017.

# Build docker image
```
docker buildx build -t network ./src/python/
```
# Start Scaphandre
To start Scaphandre and Prometheus you should use
```
cd compose
docker-compose up -d
```
Grafana can be accessed on http://localhost:3000

# Run the network in docker
To run the training and measurement process in docker you should use run_network.h $1 $2 $3 $4 $5
$1 - the size of the dataset to be used
$2 - [True False] if the smaller network should be used
$3 - numbet of epochs
$4 - [True False] if the pytorch automatic mixed precision should be used
$5 - the size of the batch 
```
./run_network.sh 60000 False 5 False 32
``` 
If at least one of the arguments is not present or has wrong format, the training will be started with default arguments of (30000 False 2 False 32)
