#!/bin/bash

env="source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate"
name=$(hostname)

if ! grep -q "$env" ~/.bashrc; then
    echo $env | sudo tee -a ~/.bashrc > /dev/null
fi

source ~/.bashrc

if [ ! -f influxdata-archive_compat.key ]; then
    wget -q https://repos.influxdata.com/influxdata-archive.key
    gpg --no-default-keyring --show-keys --with-fingerprint --with-colons ./influxdata-archive.key 2>&1 | grep -q '^fpr:\+24C975CBA61A024EE1B631787C3D57159FC2F927:$' && cat influxdata-archive.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive.gpg > /dev/null
    echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

    sudo apt-get update && sudo apt-get install influxctl influxdb2 influxdb2-cli -y
    sudo systemctl start influxdb

    influx setup \
      --username $name \
      --org "ucberkeley" \
      --bucket "lab6" \
      --force
fi