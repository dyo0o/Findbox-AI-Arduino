// 배선: 스크루 실드에서 같은 번호 단자에 연결한다
const int PIN_IR = 2;            // IR 센서 OUT (VCC → 5V, GND → GND)
const int PIN_BUZZER = 3;        // 능동 부저 SIG (VCC → 5V, GND → GND)
const int PIN_LED_R = 9;         // RGB LED R  (문을 닫았을 때 파랑이 켜지면 맞게 꽂힌 것)
const int PIN_LED_G = 10;        // RGB LED G
const int PIN_LED_B = 11;        // RGB LED B  (− → GND)

// 부품에 따라 다를 수 있는 값
const int DOOR_CLOSED_LEVEL = LOW;    // 문이 닫혀 IR 센서 앞을 가릴 때의 OUT 값. 거꾸로 동작하면 HIGH로
const int BUZZER_ON_LEVEL = LOW;      // 부저가 울리는 SIG 값 (SunFounder 모듈은 LOW). 켜자마자 계속 울리면 HIGH로
const bool LED_COMMON_ANODE = false;  // RGB LED의 공통 다리를 5V에 꽂는 제품이면 true

// 시간 (밀리초)
const unsigned long DEBOUNCE_MS = 100;            // 문 신호가 이만큼 그대로여야 진짜로 바뀐 것으로 본다
const unsigned long SETTLE_MS = 2000;             // 문이 닫힌 뒤 손이 빠지고 물건이 자리 잡을 때까지 기다림
const unsigned long RESPONSE_TIMEOUT_MS = 60000;  // 서버 응답을 기다리는 최대 시간 (인식 재시도 포함)
const unsigned long ERROR_SHOW_MS = 3000;         // 빨간 불을 켜 두는 시간
const unsigned long CHECK_STEP_MS = 500;          // 켤 때와 TEST에서 색 하나를 켜 두는 시간
const unsigned long BEEP_MS = 150;                // 짧은 소리 (삑)
const unsigned long BEEP_GAP_MS = 120;            // 짧은 소리 사이 쉬는 시간
const unsigned long CHIRP_MS = 50;                // 아주 짧은 소리 세 번 (삐리릭, 새 물건 없음·인식 실패)
const unsigned long LONG_BEEP_MS = 700;           // 긴 소리 (삐―, 서버가 대답하지 않을 때)

const int BUZZER_OFF_LEVEL = (BUZZER_ON_LEVEL == LOW) ? HIGH : LOW;

// 신호가 DEBOUNCE_MS 동안 그대로일 때만 바뀐 것으로 본다 (문 떨림 무시)
struct Debounced {
  bool stable = false;  // 확정된 상태
  bool last = false;    // 마지막으로 읽은 값
  unsigned long changedAt = 0;

  // 새로 읽은 값을 넣는다. 확정된 상태가 바뀐 순간에만 true
  bool update(bool value) {
    if (value != last) {
      last = value;
      changedAt = millis();
    }
    if (value != stable && millis() - changedAt >= DEBOUNCE_MS) {
      stable = value;
      return true;
    }
    return false;
  }
};

enum State { IDLE, SETTLING, WAITING, SHOW_ERROR };

Debounced door;     // stable: 문이 닫혀 있는지
State state = IDLE;
unsigned long stateStart = 0;
bool pendingDrop = false;  // 서버 결과를 기다리는 동안 문이 또 닫혔는지 (끝나면 이어서 처리)
char line[16];             // 서버에서 받는 한 줄
byte lineLength = 0;

void setLed(int pin, bool on) {
  // 공통 음극(− 다리를 GND에 연결)은 HIGH가 켜짐, 공통 양극은 LOW가 켜짐
  digitalWrite(pin, on != LED_COMMON_ANODE ? HIGH : LOW);
}

void setColor(bool red, bool green, bool blue) {
  setLed(PIN_LED_R, red);
  setLed(PIN_LED_G, green);
  setLed(PIN_LED_B, blue);
}

void buzz(unsigned long ms) {
  digitalWrite(PIN_BUZZER, BUZZER_ON_LEVEL);
  delay(ms);
  digitalWrite(PIN_BUZZER, BUZZER_OFF_LEVEL);
}

// 삑(1), 삐삐(2)
void beep(int times) {
  for (int i = 0; i < times; i++) {
    buzz(BEEP_MS);
    delay(BEEP_GAP_MS);
  }
}

// 삐리릭
void chirp() {
  for (int i = 0; i < 3; i++) {
    buzz(CHIRP_MS);
    delay(CHIRP_MS);
  }
}

// 삐―
void longBeep() {
  buzz(LONG_BEEP_MS);
}

// LED 빨강 → 초록 → 파랑을 하나씩 켜고 '삑'. print가 true면 지금 켜진 것을 시리얼 모니터에 찍는다 (TEST)
void checkParts(bool print) {
  const int pins[] = {PIN_LED_R, PIN_LED_G, PIN_LED_B};
  const char* names[] = {"RED", "GREEN", "BLUE"};
  for (int i = 0; i < 3; i++) {
    setColor(i == 0, i == 1, i == 2);
    if (print) {
      Serial.print(names[i]);
      Serial.print(" D");
      Serial.println(pins[i]);
    }
    delay(CHECK_STEP_MS);
  }
  setColor(false, false, false);
  if (print) {
    Serial.print("BUZZER D");
    Serial.println(PIN_BUZZER);
  }
  beep(1);
}

void enter(State next) {
  state = next;
  stateStart = millis();
  if (next == IDLE) {
    setColor(false, true, false);   // 초록
  } else if (next == SHOW_ERROR) {
    setColor(true, false, false);   // 빨강
  } else {
    setColor(false, false, true);   // 파랑 (SETTLING, WAITING)
  }
}

void backToIdle() {
  enter(IDLE);
  if (pendingDrop && door.stable) {  // 기다리는 동안 또 넣고 닫았으면 바로 이어서 처리
    enter(SETTLING);
  }
  pendingDrop = false;
}

void onDoorClosed() {
  if (state == IDLE) {
    enter(SETTLING);
    beep(1);
  } else if (state == WAITING || state == SHOW_ERROR) {
    pendingDrop = true;  // 지금 것이 끝나면 이어서 처리한다
    beep(1);
  }
}

void onDoorOpened() {
  if (state == SETTLING) {
    enter(IDLE);  // 2초 안에 다시 열면 취소하고, 다시 닫으면 처음부터
  }
}

// 한 줄이 다 들어오면 처리한다
void processLine() {
  if (state == IDLE && strcasecmp(line, "TEST") == 0) {
    checkParts(true);
    enter(IDLE);
  } else if (state == WAITING && strcasecmp(line, "OK") == 0) {
    beep(2);
    backToIdle();
  } else if (state == WAITING && strcasecmp(line, "FAIL") == 0) {
    enter(SHOW_ERROR);  // 새 물건 없음·인식 실패 (물건을 꺼내고 닫아도 이렇게 된다)
    chirp();
  }
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      line[lineLength] = '\0';
      if (lineLength > 0) {
        processLine();
      }
      lineLength = 0;
    } else if (c != '\r' && lineLength < sizeof(line) - 1) {
      line[lineLength++] = c;
    }
  }
}

void setup() {
  digitalWrite(PIN_BUZZER, BUZZER_OFF_LEVEL);  // 출력으로 바꾸기 전에 꺼 둔다 (켤 때 잡음 방지)
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_IR, INPUT);
  pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_LED_G, OUTPUT);
  pinMode(PIN_LED_B, OUTPUT);
  Serial.begin(9600);

  checkParts(false);  // 배선 확인: 빨강 → 초록 → 파랑 → '삑'

  // 켤 때 이미 닫혀 있는 문은 '닫힘'으로 치지 않는다
  door.stable = door.last = (digitalRead(PIN_IR) == DOOR_CLOSED_LEVEL);
  enter(IDLE);
}

void loop() {
  readSerial();

  if (door.update(digitalRead(PIN_IR) == DOOR_CLOSED_LEVEL)) {
    if (door.stable) {
      onDoorClosed();
    } else {
      onDoorOpened();
    }
  }

  unsigned long elapsed = millis() - stateStart;
  if (state == SETTLING && elapsed >= SETTLE_MS) {
    Serial.println("DROP");
    enter(WAITING);
  } else if (state == WAITING && elapsed >= RESPONSE_TIMEOUT_MS) {
    enter(SHOW_ERROR);
    longBeep();  // 서버가 대답하지 않을 때만 소리로도 알린다
  } else if (state == SHOW_ERROR && elapsed >= ERROR_SHOW_MS) {
    backToIdle();
  }
}
