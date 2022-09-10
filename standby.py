import Jetson.GPIO as GPIO
import time

from IIELab.GroveStartKit.GroveStarterKit import TouchSensor
from IIELab.GroveStartKit.GroveStarterKit import LEDBar
PIN = 33
clk = 36
data = 31
GPIO.setmode(GPIO.BOARD)
touch_close = TouchSensor()
touch_close.attach(PIN)
ledBar = LEDBar(clk, data)
ledBar.setLevel(1)
print('Now in standby mode...')

try:
    time.sleep(3) # To prevent accidental touch
    while True:
         
        if touch_close.isTouched():
            print('Activate!')
            break
        else: 
            time.sleep(0.1)


except KeyboardInterrupt:
    print('Terminate standby mode.')
    exit()
