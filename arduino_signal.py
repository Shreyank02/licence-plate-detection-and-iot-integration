import serial
import time

def send_open_signal():
    try:
        arduino = serial.Serial('COM5', 9600, timeout=1)
        time.sleep(2) 

        arduino.write(b'OPEN')   
        print("Signal sent to Arduino.")

        arduino.close()
    except Exception as e:
        print("Error communicating with Arduino:", e)
