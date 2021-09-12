# Prediciting Transaction Targets in Banking Data

The Dataset is obtained courtesy of Santander as a part of the Kaggle competition <a href = https://www.kaggle.com/c/santander-customer-transaction-prediction>Santander Customer Transaction Prediction</a>. The main goal of this competition was to predict "which customers will make a specific transaction in the future".  

## Table of Contents
1. [Introduction](#Introduction)
2. [About the Data](#Data)
3. [Installation & Usage](#install)
4. [Future Work](#Future-Work)


## Introduction <a name="Introduction"></a>

This Project aims to solve the stated business problem with the help of the XGBoost algorithm. The overall approach to this problem 

## About the Data <a name="Data"></a>

The Data consists of 200 anonymized features, a ID code column and a binary taget variable. The data is already provisioned in the format of a train and test csv files.


## Installation & Usage <a name="install"></a>

Follow the following instructions to setup the working directory on your local machine(**OS - Ubuntu**). It is highly recommended to use a virtual environment to run the application in order to avoid disrupting global package configuration on your machine.

```
git clone https://github.com/akhilnas/santander-customer-transaction.git
pipenv shell
pipenv install -r requirements.txt
```

If instead you would like to use the docker image(<a href=https://hub.docker.com/repository/docker/akhiln/santander/general>Link</a>) and run the container on your machine(due to difference in Operating System), the instructions are as follows.

**Note:** To install Docker please refer to this  <a href=https://docs.docker.com/get-docker/>website</a>.

```
docker pull akhiln/santander:1.0
docker run --gpus all -it -p 5000:5000 akhiln/santander:1.0
```
The container is built as per the specification of using all local available gpus, the port 5000 is mapped to the respective container port (MLFlow UI) and the -it flag tells docker that it should open an interactive container instance. To edit this command as per requirements please refer to Docker docs.

The two important files are *training.py* and *hyperparameter.py*. The training module exists in the first file while the hyperparamter tuning using Optuna is performed in the second file.

The *predict.py* file is used to generate predictions using on the saved model.


## Future Work <a name="Future-Work"></a>

The Project currently requires either a powerful local machine on the part of the user or requires the usage of cloud compute resources which leads to more manual work. Also there is many instances along this project lifecycle that human intervention is required.

In order to automate the whole pipeline and to keep in line with MLOps principles I in the future intend to set up a entire ML pipeline with the help of Amazon Sagemaker to fully automate the building, training, testing and deployment of the model. Upon each push to the Git repository the pipeline would be triggered and the entire process can be automated. Another positive would be to reduce the infrastructure management work needed as AWS will be taking care of most of work in the background.









