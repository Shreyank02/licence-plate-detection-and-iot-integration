#include <Servo.h>

Servo gateServo;

const int servoPin = 9; 
const int openAngle = 90;   
const int closeAngle = 0; 
const int delayBeforeClose = 4000; 

void setup() {
  Serial.begin(9600);
  gateServo.attach(servoPin); 
  gateServo.write(closeAngle); 
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();  

    if (command == "OPEN") {
      openGate();
    }
  }
}

void openGate() {
  gateServo.write(openAngle);       
  delay(delayBeforeClose);          
  gateServo.write(closeAngle);      
}
