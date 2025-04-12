#include <WiFiS3.h>

char ssid[] = "Rachit";     
char pass[] = "03062023"; 

WiFiServer server(80);

String instruction;
int n;
int i = 0;
#define IN1 8
#define IN2 9
#define ENA 10  // PWM

// Define L298N Motor B pins
#define IN3 11
#define IN4 12
#define ENB 13  // PWM-

int speed = 128;

void forward()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, speed); // Max speed
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, speed); // Max speed
  return;
}
void left()
{
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, speed); // Max speed
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, speed); // Max speed
  return;
}

void right()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, speed); // Max speed
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENB, speed); // Max speed
  return;
}
void stop()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, speed); // Max speed
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, speed); // Max speed
  delay(100);
  return;
}

void run()
{
  i = 0;
  while(i < n)
  {
   if(instruction[i] == 'r')
   {
     right();
     delay(700);
     stop();
     //stop
   }
   else if(instruction[i] == 'f')
   {
     forward();
     delay(425);
     stop();
     //stop
   }
   else if(instruction[i] == 'l')
   {
     left();
     delay(700);
     stop();
     //stop
   }
   i++;
  }
  stop();
}

void setup() {
  Serial.begin(9600);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    delay(1000);
    Serial.println("Trying to connect...");
  }

  Serial.print("Connected! IP Address: ");
  Serial.println(WiFi.localIP());
  server.begin();

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
        WiFiClient client = server.available();
        if (client) {
          instruction = client.readStringUntil('\r');
          n = instruction.length();
          Serial.print("Received: ");
          Serial.println(instruction);
          Serial.println(n);
          // Send response
          client.println("HTTP/1.1 200 OK");
          client.println("Content-Type: text/plain");
          client.println("Connection: close");
          client.println();
          client.println("String received!");
          delay(10);
          client.stop();
          if(instruction.length() > 0)
          {
            run();
          }  // Close connection
        }
        // if (Serial.available()) {
        //     while (Serial.available()) {
        //         char c = Serial.read();
        //         instruction += c;
        //         n++;
        //         delay(10); // Small delay to ensure full message reception
        //     }
        //     Serial.print("Received: "); 
        //     Serial.println(instruction);
        // }
    }
}
void loop() {
while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    delay(1000);
    Serial.println("Trying to connect...");
  }

  Serial.print("Connected! IP Address: ");
  Serial.println(WiFi.localIP());
  server.begin();

  instruction = "";
  n = 0;
       while (n == 0) { 
         // Keep waiting until data is received
        WiFiClient client = server.available();
        if (client) {
          instruction = client.readStringUntil('\r');
          n = instruction.length();
          Serial.print("Received: ");
          Serial.println(instruction);
          Serial.println(n);
          // Send response
          client.println("HTTP/1.1 200 OK");
          client.println("Content-Type: text/plain");
          client.println("Connection: close");
          client.println();
          client.println("String received!");
          delay(10);
          client.stop();
          if(instruction.length() > 0)
          {
            run();
          }  // Close connection
        }
        // if (Serial.available()) {
        //     while (Serial.available()) {
        //         char c = Serial.read();
        //         instruction += c;
        //         n++;
        //         delay(10); // Small delay to ensure full message reception
        //     }
        //     Serial.print("Received: "); 
        //     Serial.println(instruction);
        // }
    }

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