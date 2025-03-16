
String instruction = "rfflfrflfflfflf";
int n = sizeof(instruction)/sizeof(char);
int i = 0;
void setup() {
  
}

void loop() {
  while(i < n)
  {
    if(instruction[i] == 'r')
    {
      //right turn
      delay(1000);
      //stop
      i++;
    }
    else if(instruction[i] == 'f')
    {
      //forward code
      delay(1000);
      //stop
      i++;
    }
    else if(instruction[i] == 'l')
    {
      //left turn
      delay(1000);
      //stop
      i++;
    }
  }
}
