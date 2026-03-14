# Lab8_measure_carbon_footprint
Lab8 using Code Carbon Lib

## Create and activate Env
python3 -m venv ~/AI_Society_env

source  ~/AI_Society_env/bin/activate

# install CodeCarbon

pip install codecarbon

# Configure CodeCarbon

Follow: https://docs.codecarbon.io/latest/getting-started/usage/

Config: 
codecarbon config

Login:
codecarbon login

Detect Hardware:
codecarbon detect

## To code carbon code: 
codecarbon monitor -- bash run_experiments.sh 0.001,0.01 5 

