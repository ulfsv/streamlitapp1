# version 2020-09-29
# changed imbalance to false as default
import streamlit as st
import matplotlib.pyplot as plt
from helpers2 import load_data, describe_sample, createmodel, createmodelmcc, predict, unseendata, final, savemodel
import os 
import pycaret.classification as cl
import pandas as pd

#from google.cloud import storage
#st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

def main():   
   
#st.subheader("Login Section")
#login = st.button("Login") 
#if login:
#username = st.sidebar.text_input("User Name")
#password = st.sidebar.text_input("Password",type='password')

#login = st.sidebar.button("Login") 
#if login:
#if st.sidebar.checkbox("Login"):
#    create_usertable()
#    hashed_pswd = make_hashes(password)
#    result = login_user(username,check_hashes(password,hashed_pswd))

    #   if result:	
        #st.success('Logged in as: {}'.format(username))

        global model 
        global tuned_model 
        
        st.sidebar.markdown('Upload training data as a csv file with utf-8 encoding. The data is cleaned: Duplicates are dropped. Missing values are replaced with the mode (for categorical variables) or median (for continuous variables) on a column-by-column basis. Non-numerical variables are encoded (e.g., categorical variables with strings) with numerical equivalents. Target column used for training is Class. ')     
        link = '[Test dataset to download](https://drive.google.com/file/d/1fcrUwR-UerSPK80_RXXGcFrz7A4rMzMS/view?usp=sharing)'
        st.sidebar.markdown(link, unsafe_allow_html=True)
        file_upload = st.sidebar.file_uploader("Upload csv file for predictions", type="csv")
        
        b = st.sidebar.slider("Percent to use for training", 0, 90, 80)  # 👈 Changed this
        percent = (b/100)
        st.sidebar.markdown('Train the model with stratified cross validation. Select number of folds to use. Reducing the number of folds will improve the training time.')
        numfolds = st.sidebar.slider("Number of folds", 0, 20, 7)  
        numfolds = int(numfolds)
        #st.write("Percent:", b)
        #st.write("Percent:", percent)
        #st.sidebar.info('This app is created to predict credit fraud')

        st.title("Train and evaluate a model for Credit fraud Prediction")

        if file_upload is not None:
            data = load_data(file_upload)   
            # kolla storleken på filen
            # ställ in konfigurering beroende på storlek på filen antal rader och kolumner
            # eller hur stor filen är i mb ?
            # om filen är < 5000 rader, 20 kolumner kör max inställningar
            # över 5000 < 10 000 använd endel av inställningarna       
            #st.write('Data for Modeling: ' + str(data_train.shape)) 
            #st.write(data)
            st.markdown('''### Cleaned dataset''')
            size = data.memory_usage(deep=True).sum()
            st.markdown('''size in Mb''')
            st.write(size/1000000)
            st.dataframe(data.head(10)) 
            datashape = ('Cleaned dataset (rows, columns): ' + str(data.shape))
            st.write(datashape) 
            
            described_sample = describe_sample(data)  # input into streamlit object: ✅
            st.markdown('''### Describe the dataset''')
            st.write(described_sample)

            normalise = st.sidebar.radio("Normalize the data?", ('Yes', 'No'))
            if normalise == 'Yes':
                opti = True 
            else:
                opti = False

            normalisemet = st.sidebar.radio("Normalize method", ('zscore', 'minmax', 'maxabs', 'robust'))           

            tranzform = st.sidebar.radio("Transform the data? A power transformation is applied to make the data more normal / Gaussian-like", ('Yes', 'No'))
            if tranzform == 'Yes':
                tranz = True 
            else:
                tranz = False

            lowvar = st.sidebar.radio("Ignore low variance?", ('Yes', 'No'))
            if lowvar == 'No':
                inglowvar = False 
            else:
                inglowvar = True

            imbalance = st.sidebar.radio("Fix imbalance?", ('Yes', 'No'))
            if imbalance == 'No':
                imb = False 
            else:
                imb = True

            featur = st.sidebar.radio("Feature interaction. Create new features by interacting (a * b) for all numeric variables in the dataset including polynomial and trigonometric features", ('No', 'Yes'))
            if featur == 'No':
                featureinter = False 
            else:
                featureinter = True

            featureratio = st.sidebar.radio("Feature ratio? Create new features by calculating the ratios (a / b) of all numeric variables in the dataset.", ('No', 'Yes'))
            if featureratio == 'No':
                fratio = False 
            else:
                fratio = True 

            polynom = st.sidebar.radio("Create polynominal features?", ('No', 'Yes'))
            if polynom == 'No':
                polyf = False 
            else:
                polyf = True   

            polydeg = st.sidebar.slider("Degree of features (if degrees is more than 2 set Create polynominal features = No to avoid errors)", 0, 5, 2)     
            polydeg = int(polydeg)
            trigo = st.sidebar.radio("Create trigonometric features?", ('No', 'Yes'))
            if trigo == 'No':
                trigf = False 
            else:
                trigf = True  

            futureselect = st.sidebar.radio("Drop features based on permutation importance techniques including Random Forest, Adaboost and Linear correlation with target variable? Default treshold 0.8.", ('Yes', 'No'))
            if futureselect == 'Yes':
                futsel = True 
            else:
                futsel = False

            tunemodel = st.sidebar.radio("Tune the model?", ('No', 'Yes'))

            if st.checkbox("Train a model"):
                exp_clf102 = cl.setup(data = data, target = 'Class', session_id=123,
                            html = False,                        
                            normalize = opti, 
                            normalize_method = normalisemet,     
                            transformation = tranz,
                            polynomial_features = polyf,
                            polynomial_degree = polydeg, 
                            #polynomial_degree = 4,
                            trigonometry_features = trigf,
                            ignore_low_variance = inglowvar,
                            remove_multicollinearity = True,
                            multicollinearity_threshold = 0.9, 
                            train_size = percent,
                            profile = False,
                            fix_imbalance = imb, 
                            feature_selection = futsel,
                            feature_interaction = featureinter,
                            feature_ratio = fratio,
                            #sampling=False, 
                            silent=True,
                            log_experiment = False,
                            log_data = False
                            )

                if tunemodel == 'Yes':
                    model = createmodel(numfolds)
                    model = createmodelmcc(model, numfolds)            
                else:              
                    model = createmodel(numfolds)
                    st.markdown('''### Model performance''')
                    st.markdown('The model are scored using stratified cross validation on the test set and evaluated with the 7 most commonly used classification metrics (Accuracy, AUC, Recall, Precision, F1, Kappa and MCC).')       
                    compare_cv_results = (cl.get_config('display_container')[-1])
                    st.dataframe(compare_cv_results) 

                    
                
            st.markdown('''### Transformed dataset''')
            st.markdown('''When clicking the transformed dataset button the table shows the data after it has been transformed according to the configuration in the left sidebar. This shows the data that will be split in to train, test and an hold out set''')
                
            if st.checkbox("Look at the Transformed dataset"):
                transformed_data = cl.get_config('X')     
                st.dataframe(transformed_data.head(10)) 
                shape = ('Transformed data (rows, columns): ' + str(transformed_data.shape))
                st.write(shape)

            button_aucroc = st.button("Plot AUC-ROC curve")     

            st.markdown('''### AUC-ROC Curve on test set''')
            st.markdown('''AUC-ROC curve is a performance measurement for classification problem at various thresholds settings.  ROC, Receiver Operating Characteristic curve is a probability curve and AUC, Area Under Curve represents degree or measure of separability.  It tells how much model is capable of distinguishing between classes.  Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.    The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.''')    
            
            if button_aucroc:
                #fig = cl.plot_model(model, plot='auc')
                #st.pyplot(plot_model(model, plot='auc'))
                #st.pyplot(fig)
                fig = cl.plot_model(model, plot = 'auc', save=True)
                st.image('./AUC.png')

            button_pr = st.button("Plot PR curve")     
            
            st.markdown('''### Precision-Recall Curve''')
            st.markdown('''Precision is a ratio of the number of true positives divided by the sum of the true positives and false positives. It describes how good a model is at predicting the positive class on the test set. Precision is referred to as the positive predictive value. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. Specifically, there are many examples of no event (class 0) and only a few examples of an event (class 1).''')
            
            if button_pr:
                xgbpr = cl.plot_model(model, plot='pr', save=True)                    
                st.image('./Precision Recall.png')

            button_confusion = st.button("Plot Confusion matrix")            
            if button_confusion:             
                xgbconf = cl.plot_model(model, plot='confusion_matrix', save=True)
                st.image('./Confusion Matrix.png')

            button_classerr = st.button("Plot Class Prediction Error")            
            if button_classerr:  
                st.markdown('''## Class Prediction Error on test set''')           
                err = cl.plot_model(model, plot='error', save=True)
                st.image('./Prediction Error.png')

            button_learning = st.button("Plot Learning curve")            
            if button_learning:             
                learning = cl.plot_model(model, plot='learning', save=True)
                st.image('./Learning Curve.png')

            button_feature = st.button("Plot Feature importance")            
            if button_feature:             
                learning = cl.plot_model(model, plot='feature', save=True)
                st.image('./Feature Importance.png')

            button_unseen = st.button("Predict on unseen data")            
            if button_unseen: 
                data_unseen = unseendata(data)
                #unseen_predictions=""
                unseen_predictions = predict(model, data_unseen)
                st.markdown('''## Predict on unseen data''')
                st.markdown('This is 10% of the original data which was never exposed to training or validation')
                st.markdown('The last three columns shows: the original Class flagged for fraud 1, or no fraud 0, Label is the prediction of the class. Score is probability of positive outcome.')
                st.dataframe(unseen_predictions) 
                shape = ('Unseen data (rows, columns): ' + str(unseen_predictions.shape))
                st.write(shape)             
                savedfromfraud = unseen_predictions.loc[unseen_predictions['Label'] == '1', 'amount'].sum()
                savedamount = ('Amount saved (USD): ' + str(savedfromfraud))
                st.write(savedamount) 
                #st.write(savedfromfraud) 
                #savedfromfraud 

            button_finalize = st.button("Save finalized model")
            if button_finalize:
                link = savemodel(model)
                #st.write(saved_model)
                #import time
                #timestr = time.strftime("%Y%m%d-%H%M%S")
                #File_name = 'prod'
                #directory_to_save_to = 'models'  # change this if the folder has another name
                #name_of_file_to_save = File_name + '_new_' + timestr
                #cl.save_model(model, name_of_file_to_save)
                #st.write(cl.save_model(model, name_of_file_to_save))

                st.markdown('Download the finalized model')     
                link2 = '[Test dataset to download](link)'
                st.markdown(link, unsafe_allow_html=True)

        #else:
        #    st.warning("Incorrect Username/Password")

if __name__ == '__main__':
    main()
