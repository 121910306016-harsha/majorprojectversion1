from symtable import symtable
from flask import Flask  
import string
import numpy as np
import pandas as pd
from collections import Counter
from pandas import DataFrame, read_csv;
import sklearn  
from flask import Flask, render_template, request, redirect, url_for   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import StratifiedKFold
from ecgdetectors import Detectors
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk
import biosppy
import tensorflow as tf
from keras.models import load_model 
import os
from werkzeug.utils import secure_filename
app = Flask(__name__) 
def DecisionTree(s1,s2,s3,s4,s5):
    result=''
    from sklearn import tree
    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X.values,y)
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    psymptoms = [s1,s2,s3,s4,s5]

    for k in range(0,len(symtoms)):
        for z in psymptoms:
            if(z==symtoms[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]
    p2=clf3.predict_proba(inputtest)
    print(p2)

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        result=disease[a]
    else:
        result="Not Found"
    print(result)
    return result
def randomforest(s1,s2,s3,s4,s5):
    result=''
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X.values,np.ravel(y))
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    psymptoms = []
    psymptoms.append(s1)
    psymptoms.append(s2)
    psymptoms.append(s3)
    psymptoms.append(s4)
    psymptoms.append(s5)
    
    for k in range(0,len(symtoms)):
        for z in psymptoms:
            if(z==symtoms[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]
    p2=clf4.predict_proba(inputtest)
    print(p2)

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        result=disease[a]
    else:
        result="Not Found"
    print(result)
    return result
def NaiveBayes(s1,s2,s3,s4,s5):
    result=''
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB() 
    gnb=gnb.fit(X.values,np.ravel(y))
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    psymptoms = []
    psymptoms.append(s1)
    psymptoms.append(s2)
    psymptoms.append(s3)
    psymptoms.append(s4)
    psymptoms.append(s5)
    for k in range(0,len(symtoms)):
        for z in psymptoms:
            if(z==symtoms[k]):
                l2[k]=1
    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        result=disease[a]
    else:
        result="Not Found"
    print(result)
    return result
def Logistic(s1,s2,s3,s4,s5):
    result=''
    from sklearn.linear_model import LogisticRegression
    clf5 = LogisticRegression()
    clf5 = clf5.fit(X.values,y)
    from sklearn.metrics import accuracy_score
    y_pred=clf5.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    psymptoms = [s1,s2,s3,s4,s5]
    for k in range(0,len(symtoms)):
        for z in psymptoms:
            if(z==symtoms[k]):
                l2[k]=1
    inputtest = [l2]
    predict = clf5.predict(inputtest)
    predicted=predict[0]
    p2=clf5.predict_proba(inputtest)
    print(p2)

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        result=disease[a]
    else:
        result="Not Found"
    print(result)
    return result
disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']    
symtoms=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']
l2=[]
for x in range(0,len(symtoms)):
    l2.append(0)
df=pd.read_csv("Training.csv")
df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
X= df[symtoms]
y = df[["prognosis"]]
np.ravel(y)
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
X_test= tr[symtoms]
y_test = tr[["prognosis"]]
np.ravel(y_test)
@app.route("/")               
def main(): 
    return render_template("home.html")
@app.route("/select")
def select():
    return render_template("index.html",d=symtoms)
@app.route("/predict",methods=['POST'])
def result():
    result=[]
    s1=request.form['s1']
    print(s1)
    s2=request.form['s2']
    print(s2)
    s3=request.form['s3']
    print(s3)
    s4=request.form['s4']
    print(s4)
    s5=request.form['s5']
    result.append(DecisionTree(s1,s2,s3,s4,s5))
    result.append(randomforest(s1,s2,s3,s4,s5))
    result.append(NaiveBayes(s1,s2,s3,s4,s5))
    print(result)
    s=set(result)
    return render_template("result.html",prediction=result,k=s)
#Major Project code---------------------------------------------------------------------------------------------------
final_list_X = []
mi=0
@app.route("/ecg")
def ecg():
    return render_template("ecgupload.html",mi=mi)
def filter_ecg(val):
    ecg = val 
    Fs = 500 
    N = ecg.shape[1]
    t = ((np.linspace(0, N-1, N))/(Fs))
    cover = t.shape[0]
    t = t.reshape(1, cover)  
    n=2 
    Fcutoff_low = 0.5 
    Wn_low = ((2*Fcutoff_low)/(Fs))
    b_low, a_low = signal.butter(n, Wn_low, 'low')
    xn_filtered_LF = signal.filtfilt(b_low, a_low, ecg)
    Fcutoff_high = 40 
    Wn_high = ((2*Fcutoff_high)/(Fs))
    b_high, a_high = signal.butter(n, Wn_high, 'high')
    xn_filtered_HF = signal.filtfilt(b_high, a_high, ecg)
    xn = (ecg-xn_filtered_HF-xn_filtered_LF)
    return xn
def return_features(ecg_test):
    cleaned = nk.ecg_clean(ecg_test, sampling_rate = 500)  
    rdet, = biosppy.ecg.hamilton_segmenter(signal = cleaned, sampling_rate = 500)   
    rdet, = biosppy.ecg.correct_rpeaks(signal = cleaned, rpeaks = rdet, sampling_rate = 500, tol = 0.05)
    if(rdet.size<=4):       
        return 'INCOMPLETE'
    rdet = np.delete(rdet, -1)       
    rdet = np.delete(rdet, 0)
    rpeaks = {'ECG_R_Peaks': rdet}   
    cleaned_base = nk.signal_detrend(cleaned, order=0)
    signals, waves = nk.ecg_delineate(cleaned_base, rpeaks, sampling_rate = 500, method = "dwt") 
    rpeakss = rpeaks.copy() 
    temppo = 4-len(rpeakss['ECG_R_Peaks'])
    if temppo>0:
        for i in range(temppo):
            rpeakss['ECG_R_Peaks'] = np.append(rpeakss['ECG_R_Peaks'], rpeakss['ECG_R_Peaks'][-1] + 1)
    signals1, waves1 = nk.ecg_delineate(cleaned_base, rpeakss, sampling_rate = 500, method = "peak")
    if temppo>0:
        for j in range(temppo):
            waves1['ECG_Q_Peaks'] = waves1['ECG_Q_Peaks'][:-1] 
    return (cleaned_base, [waves['ECG_T_Peaks'], waves['ECG_R_Onsets'], waves['ECG_R_Offsets'], waves1['ECG_Q_Peaks']])
def result_array(given_list):
    final_list_X=[]
    mini = 50; 
    for check_index3 in range(12):
        for second_ind3 in range(4):
            mini = min(mini, len(given_list[check_index3][1][second_ind3]))
    to_take = min(16, mini)
    for x in range(to_take):
        a_temp_list = []  
        flag = -1      
        for y in range(12): 
            if((np.isnan(given_list[y][1][1][x])) or (np.isnan(given_list[y][1][2][x])) or (np.isnan(given_list[y][1][3][x])) or (np.isnan(given_list[y][1][0][x]))):
                a_temp_list = []
                flag = 1
                break 
            first_feat = given_list[y][0][int(given_list[y][1][1][x])] - given_list[y][0][int(given_list[y][1][2][x])]
            second_feat = given_list[y][0][int(given_list[y][1][3][x])]  
            third_feat = given_list[y][0][int(given_list[y][1][0][x])]
            a_temp_list.append(first_feat)
            a_temp_list.append(second_feat)
            a_temp_list.append(third_feat)
        if(flag == -1):
            final_list_X.append(a_temp_list)
    return final_list_X
def read_data(head):
    signal, meta_val = wfdb.rdsamp('C:/Users/chint/OneDrive/Desktop/sample/Projectversion1/uploaded/'+head)   
    value = signal.T
    temp_list = []
    flag1 = -1
    for ind in range(12):
        val_ind = value[ind]
        tmpp = val_ind.shape[0]
        val_ind = val_ind.reshape(1, tmpp)
        val_filtered = filter_ecg(val_ind)
        val_filtered = val_filtered.reshape(val_filtered.shape[1], ) 
        a_var = return_features(val_filtered)
        if(a_var == 'INCOMPLETE'):
            temp_list = []
            flag1 = 1
            break
        temp_list.append(a_var)
    if(flag1==-1):
        f=result_array(temp_list)
    return f
# @app.route('/upload', methods=['GET', 'POST'])
def predict_mi(head):
    print("for record"+head)
    # head = request.args.get('head')
    model = load_model('C:/Users/chint\OneDrive/Desktop/sample/Projectversion1/models/lstm_model.h5')
    print("Prediction Of ECG") 
    feature_list=read_data(head)
    pred=np.argmax(model.predict(feature_list), axis = -1)
    print(pred)
    mi=1
    i=pred[0]
    if(i==0):
        temp = 'ALMI'
    elif(i==1):
        temp = 'AMI'
    elif(i==2):
        temp = 'ASMI'
    elif(i==3):
        temp = 'ILMI'
    elif(i==4):
        temp = 'IMI'
    elif(i==5):
        temp = 'IPLMI'
    elif(i==6):
        temp = 'IPMI'
    elif(i==7):
        temp = 'LMI'
    elif(i==8):
        temp = 'NORM'
    else:
        temp = 'PMI'
    return i,mi,temp
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD ='C:/Users/chint/OneDrive/Desktop/sample/Projectversion1/uploaded'
ALLOWED_EXTENSIONS = {'dat', 'hea'}
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        dat_file = request.files['dat_file']
        hea_file = request.files['hea_file']
        filename=dat_file.filename
        head=filename[:-4]
        print(2)
        dat_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dat_file.filename)))
        hea_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(hea_file.filename)))
        i,mi,temp=predict_mi(head)
    # return redirect(url_for('/predict', head=head))
    # return render_template("ecgupload.html",mi=mi)
    # if request.method == 'POST':
    #     if 'file' not in request.files:
    #         return redirect(request.url)
    #     file = request.files['file']
    #     print(2,file.filename)
    #     if file.filename == '':
    #         return redirect(request.url)
    #     if file:
    #         filename = file.filename
    #       
    #         print(2,head)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
    #         predict_mi(head)
    return render_template("ecgupload.html",res=i,mi=mi,local=temp,i=i)
    
if __name__ == "__main__":        
    app.run(debug=True)                 