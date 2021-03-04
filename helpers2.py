## helpers.py
# version 2020-10-01
# train with gpu
import streamlit as st
import pandas as pd
#from pycaret.classification import *
import pycaret.classification as cl
#from shap import *
#import mlflow
#mlflow.set_tracking_uri('file:/c:/users/mlflow-server')
#from pycaret.datasets import get_data
import numpy as np # linear algebra
#from datacleaner import autoclean
import hashlib
import sqlite3 
import time

# Security
#passlib,hashlib,bcrypt,scrypt

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management

conn = sqlite3.connect('data.db', check_same_thread=False)
#conn = sqlite3.connect('/content/gdrive/My Drive/Colab Notebooks/fraud/data.db', check_same_thread=False)
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

@st.cache
def load_data(datainput):
   
    dataset = pd.read_csv(datainput)
    dataset.drop_duplicates(keep=False, inplace=True)     
    #dataset = autoclean(dataset)
    data_train = dataset.sample(frac=0.90, random_state=786)
    data_train.reset_index(drop=True, inplace=True)
    #data_train = data_train[1:10000]

    return data_train

@st.cache
def describe_sample(dataset):
    sample = dataset
    return sample.describe(include='all')

#@st.cache
def unseendata(data):
    data_train = data.sample(frac=0.90, random_state=786)
    data_unseen = data.drop(data_train.index).reset_index(drop=True)
    return data_unseen

#@st.cache
#def describe_sample(dataset, nrows):
#    sample = dataset.head(nrows)
#    return sample.describe()


model=""
filename=""

def createmodel(fold):
    cl.set_config('html_param', 1)
    global model 
    #model = create_model('xgboost', tree_method = 'gpu_hist', gpu_id = 0, fold = fold, verbose=False)
    model = cl.create_model('xgboost', fold = fold, verbose=False)
    return model

# change html_param


def createmodelmcc(model, fold):
    cl.set_config('html_param', 1)
    #set_config('display_container', 1)  
    global tuned_model 
    tuned_model = cl.tune_model(model, optimize = 'MCC', fold = fold, verbose=False)
    return tuned_model


predictions=""

def predict(model, data):
    cl.set_config('html_param', 1)
    cl.set_config('display_container', 1)  
    #global model
    predictions = cl.predict_model(model, data=data)
    return predictions


def final(tuned_model):
    final_xgb = cl.finalize_model(tuned_model)
    return final_xgb

def savemodel(finalmodel):
    # Save the model with a timestamp
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    File_name = 'prod'
    #directory_to_save_to = 'models'  # change this if the folder has another name
    name_of_file_to_save = './'+ File_name + '_new_' + timestr
    cl.save_model(finalmodel, name_of_file_to_save)
    link = 'https://localhost:8501/' + name_of_file_to_save + '.pkl'
    return link