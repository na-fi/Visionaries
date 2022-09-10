import os
from time import sleep

try:
    while True:
        os.system("python3 standby.py")
        sleep(0.2)
        os.system("python3 sen2nano.py")
        sleep(0.2)


except KeyboardInterrupt:
    print('Terminate startup.')
