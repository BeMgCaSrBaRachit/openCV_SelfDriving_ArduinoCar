
#include <SoftwareSerial.h>

String instruction = "rfflfrflfflfflf";
int n = sizeof(instruction)/sizeof(char);
int i = 0;
#define IN1 8
#define IN2 9
#define ENA 10  // PWM

// Define L298N Motor B pins
#define IN3 11
#define IN4 12
#define ENB 13  // PWM-



void forward()
{
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 128); // Max speed
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 128); // Max speed
  return;
}
void left()
{
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 128); // Max speed
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENB, 128); // Max speed
  return;
}

void right()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, 128); // Max speed
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 128); // Max speed
  return;
}
void stop()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 128); // Max speed
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 128); // Max speed
  return;
}

void setup() {

  Serial.begin(9600);  // Serial monitor // Bluetooth module baud rate
  Serial.println("Waiting for Bluetooth data...");

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  instruction = "";
  n = 0;
       while (n == 0) { 
         // Keep waiting until data is received
        if (Serial.available()) {
            while (Serial.available()) {
                char c = Serial.read();
                instruction += c;
                n++;
                delay(10); // Small delay to ensure full message reception
            }
            Serial.print("Received: "); 
            Serial.println(instruction);
        }
    }
    if(instruction.length() > 0)
    {
      run();
    }
}

void run()
{
  i = 0;
  while(i < n)
  {
   if(instruction[i] == 'r')
   {
     right();
     delay(1000);
     //stop
     i++;
   }
   else if(instruction[i] == 'f')
   {
     forward();
     delay(1000);
     //stop
     i++;
   }
   else if(instruction[i] == 'l')
   {
     left();
     delay(1000);
     //stop
     i++;
   }
  }
  stop();
}

void loop() {

}

// void setup() {
//   // Set motor control pins as outputs
// }

// void loop() {
//   // Move both motors forward at full speed
//   digitalWrite(IN1, HIGH);
//   digitalWrite(IN2, LOW);
//   analogWrite(ENA, 255); // Max speed

//   digitalWrite(IN3, HIGH);
//   digitalWrite(IN4, LOW);
//   analogWrite(ENB, 255); // Max speed

//   delay(2000); // Run for 2 seconds

//   // Stop both motors
//   digitalWrite(IN1, LOW);
//   digitalWrite(IN2, LOW);
//   digitalWrite(IN3, LOW);
//   digitalWrite(IN4, LOW);
//   delay(1000); // Pause for 1 second

//   // Move both motors in reverse at half speed
//   digitalWrite(IN1, LOW);
//   digitalWrite(IN2, HIGH);
//   analogWrite(ENA, 128); // Half speed

//   digitalWrite(IN3, LOW);
//   digitalWrite(IN4, HIGH);
//   analogWrite(ENB, 128); // Half speed

//   delay(2000); // Run for 2 seconds

//   // Stop both motors
//   digitalWrite(IN1, LOW);
//   digitalWrite(IN2, LOW);
//   digitalWrite(IN3, LOW);
//   digitalWrite(IN4, LOW);
//   delay(1000); // Pause for 1 second
// }
