#!/bin/bash

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install --yes apache2-utils
sudo apt-get install --yes software-properties-common
sudo add-apt-repository --yes ppa:deadsnakes/ppa 
sudo apt update
sudo apt install --yes python3.9
sudo apt-get install --yes python3.9-venv

update-alternatives --install /usr/bin/python python /usr/bin/python3.9 2

python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install datetime
python -m pip install requests
python -m pip install pandas
python -m pip install bs4
python -m pip install flask
python -m pip install pyspark
python -m pip install kafka-python