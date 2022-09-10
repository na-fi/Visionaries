import time
from time import sleep
import RPi.GPIO as GPIO
from IIELab.GroveStartKit.GroveStarterKit import TouchSensor
from IIELab.GroveStartKit.GroveStarterKit import LEDBar

# set led bar and touch sensor
led = 14
PIN = 18
PIN_CLOSE = 15
GPIO.setmode(GPIO.BCM)
GPIO.setup(led,GPIO.OUT)
touch = TouchSensor()
touch.attach(PIN)
touch_close = TouchSensor()
touch_close.attach(PIN_CLOSE) 

clk = 16
data = 13

ledBar = LEDBar(clk, data)
ledBar.setLevel(1)


import serial

from datetime import datetime
import yaml
import mysql.connector

import numpy as np
import pywt
import neurokit2 as nk
import math
from scipy import signal
from scipy.signal import medfilt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite

# =======parameter========
LOCATION = 'RPi_1_location'
# ========================
ser = serial.Serial ("/dev/ttyACM0", 57600)

interpreter1 = tflite.Interpreter(model_path='tflite_model_quant_int8_i.tflite')
interpreter1.allocate_tensors()

interpreter2 = tflite.Interpreter(model_path='tflite_model_quant_int8_c.tflite')
interpreter2.allocate_tensors()

db = yaml.load(open('db.yaml'), Loader=yaml.FullLoader)

mysql_connection = mysql.connector.connect(
    host=db['mysql_host'],
    user=db['mysql_user'],
    password=db['mysql_password'],
    database=db['mysql_db']
)

cursor = mysql_connection.cursor()

# push a new data
def push(id, covid, time, location):
    cursor.execute(
        'INSERT INTO tb(ID, COVID, TIME, LOCATION) VALUES (%s, %s, %s, %s)', (id, covid, time, location))
    mysql_connection.commit()

# clear table
def clear():
    cursor.execute('truncate table tb')

# fetch all data in tb
def seeall():
    cursor.execute(
        "Select people.ID, people.NAME, tb.COVID, tb.TIME, tb.LOCATION from tb, people Where tb.ID=people.ID ORDER BY tb.TIME")
    for x in cursor:
        print(x)

# delete table
def delete():
    cursor.execute('DROP TABLE tb')

# create table
def create():
    cursor.execute(
        "CREATE TABLE tb (ID int NOT NULL, COVID enum('T', 'F'), TIME datetime NOT NULL, LOCATION varchar(255) NOT NULL)")



def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def downsample(data, prehz, newhz):
    time = int(len(data)/prehz)
    st = float(prehz)/float(newhz)
    new_series = []
    for i in range(0, time):
        pre_series = data[i*prehz:(i+1)*prehz-1]
        temp = []    
        temp.append(pre_series[0])
        for j in range(1, newhz - 1):
            index = round(j * st)
            temp.append(data[index])
        temp.append(pre_series[-1])
        new_series.append(temp)
    return new_series


def evaluate_model(interpreter, test_data):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    scale, zero_point = interpreter.get_input_details()[0]['quantization']
    
    test_data = np.int8(test_data / scale + zero_point)
    
    interpreter.set_tensor(input_index, test_data)

    interpreter.invoke()

    output = interpreter.tensor(output_index)

    result = np.argmax(output()[0])
    
    return result


print("start")
ledBar.setLevel(2)
try:
    while True:
        if touch.isTouched() :
            GPIO.output(led, True)
            time.sleep(0.1)
            GPIO.output(led, False)
            ledBar.setLevel(3)
            
            #--- receiving data
            print("receiving data")
            data = []
            for k in range(6):
                received_data = ser.read()
                sleep(1)
                data_left =ser.inWaiting()
                received_data += ser.read(data_left)

                for i in range(len(received_data)-7):
                    if received_data[i] == 0xaa and received_data[i+1] == 0xaa and received_data[i+2] == 0x4:
                        temp = received_data[i+5]*256 + received_data[i+6]
                        if (received_data[i+3] + received_data[i+4] + received_data[i+5] + received_data[i+6] + received_data[i+7])%256 == 255:
                            if temp >= 32768:
                                temp -= 65536
                            data.append(temp)
                            
                ledBar.setLevel(4+k)

            #--- receiving data end

            print (len(data))
            for i in data:
                print(i, end=',')

            print("processing")

            x = [x for x in data if math.isnan(x) == False]

            #--- identify model preprocess
            filt = int(512*0.8)
            mf = medfilt(x, filt)
            data_filted = (x - mf)[int(len(x)*0.08):-int(len(x)*0.05)]

            data_denoised = denoise(data_filted.reshape(-1,))[:2560] #小波變換
            
            scaler = MinMaxScaler(feature_range=(-128, 127))
            scaler.fit(data_denoised.reshape(-1, 1))
            data_Scale = np.trunc(scaler.transform(data_denoised.reshape(-1, 1))).flatten() #歸一化

            data_downsample360 = np.array(downsample(data_Scale, 2560, 1800)).reshape(1, 1800, 1, 1)
            #--- identify model preprocess end

            #--- covid model preprocess
            s = x[0: len(x) - len(x)%25]                  # 讓每一筆變成25的倍數，多的捨棄
            down_signal = signal.resample(s, int(len(s)/2.56)) # 每一筆的長度縮短2.56倍，頻率降為200Hz
            ecg_signal = down_signal

            filt = int(190*0.8)
            Filter = medfilt(ecg_signal, filt+1)
            signal_filt = ecg_signal - Filter

            signal_denoise = denoise(signal_filt)
            
            covid_data=denoise[:370]

            scaler = MinMaxScaler(feature_range=(-128, 127))
            scaler.fit(covid_data.reshape(-1, 1))
            data_downsample200 = np.trunc(scaler.transform(covid_data.reshape(-1, 1))).reshape(1, 370, 1, 1) #歸一化

            #--- covid model preprocess end


            #--- pridict
            result_i = evaluate_model(interpreter1, data_downsample360)
            result_c = evaluate_model(interpreter2, data_downsample200)
            if result_c == 0:
                covid = 'F'
            else:
                covid = 'T'

            push(str(result_i), covid, datetime.now(), LOCATION)
            print('get data: ', result_i, covid)
            
            ledBar.setLevel(10)
            time.sleep(2)
            ledBar.setLevel(2)
        else: 
            time.sleep(0.1)
            
        #--- close
        if touch_close.isTouched():
            ledBar.setLevel(0)
            GPIO.cleanup()
            ser.close()
            serSend.close()
            serGet.close()
            print('terminate')

except KeyboardInterrupt: 
    ledBar.setLevel(0)
    GPIO.cleanup()
    
    interpreter1.close();
    interpreter2.close();
    ser.close()
    print('terminate')
