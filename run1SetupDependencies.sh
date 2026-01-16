#!/bin/bash

sudo apt-get update
sudo apt install python3.10-venv -y
sudo apt install openjdk-17-jdk -y


#sudo update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/java-17-openjdk-amd64" 10000
#sudo update-alternatives --install "/usr/bin/javac" "javac" "/usr/lib/jvm/java-17-openjdk-amd64" 10000
#sudo update-alternatives --install "/usr/bin/javaws" "javaws" "/usr/lib/jvm/java-17-openjdk-amd64" 10000
#sudo update-alternatives --config java