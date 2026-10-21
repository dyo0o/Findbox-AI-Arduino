// FindBox 보관함 아두이노 (Arduino Uno + 스크루 실드)
//
// 문이 닫히면 2초 기다렸다가 노트북 서버에 DROP을 보내고, 서버 결과(OK / FAIL)에 따라
// RGB LED와 부저로 알린다. 서버 쪽 처리는 B.md 2번 참고.
//
//   LED  초록: 대기   파랑: 촬영·인식 중   빨강: 새 물건 없음·인식 실패·서버 응답 없음 (3초 뒤 초록)
//   부저 등록이 끝나면(OK) "삐삐"
//
// 시리얼 (9600bps, 한 줄씩)
//   아두이노 → 서버: DROP (문 닫힘), RESET (보관함 비움 버튼)
//   서버 → 아두이노: OK (등록 완료), FAIL (새 물건 없음·인식 실패)

// 배선: 스크루 실드에서 같은 번호 단자에 연결한다
const int PIN_IR = 2;            // IR 센서 OUT (VCC → 5V, GND → GND)
const int PIN_BUZZER = 3;        // 능동 부저 I/O (VCC → 5V, GND → GND)
const int PIN_RESET_BUTTON = 4;  // (선택) 보관함 비움 버튼: D4와 GND 사이에 푸시 버튼. 안 달아도 된다
const int PIN_LED_R = 9;         // RGB LED R
const int PIN_LED_G = 10;        // RGB LED G
const int PIN_LED_B = 11;        // RGB LED B (− 다리 → GND)

// 부품에 따라 다를 수 있는 값
const int DOOR_CLOSED_LEVEL = LOW;    // 문이 닫혀 IR 센서 앞을 가릴 때의 OUT 값. 거꾸로 동작하면 HIGH로
const int BUZZER_ON_LEVEL = LOW;      // 부저가 울리는 I/O 값. 켜자마자 계속 울리면 HIGH로
const bool LED_COMMON_ANODE = false;  // RGB LED의 공통 다리를 5V에 꽂는 제품이면 true

// 시간 (밀리초)
const unsigned long DEBOUNCE_MS = 100;            // 신호가 이만큼 그대로여야 진짜로 바뀐 것으로 본다
const unsigned long SETTLE_MS = 2000;             // 문이 닫힌 뒤 손이 빠지고 물건이 자리 잡을 때까지 기다림
const unsigned long RESPONSE_TIMEOUT_MS = 60000;  // 서버 응답을 기다리는 최대 시간 (인식 재시도 포함)
const unsigned long ERROR_SHOW_MS = 3000;         // 빨간 불을 켜 두는 시간

const int BUZZER_OFF_LEVEL = (BUZZER_ON_LEVEL == LOW) ? HIGH : LOW;

// 신호가 DEBOUNCE_MS 동안 그대로일 때만 바뀐 것으로 본다 (문 떨림, 버튼 접점 떨림 무시)
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
Debounced button;   // stable: 버튼이 눌려 있는지
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

void beep(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(PIN_BUZZER, BUZZER_ON_LEVEL);
    delay(120);
    digitalWrite(PIN_BUZZER, BUZZER_OFF_LEVEL);
    delay(100);
  }
}

// 서버에서 한 줄이 다 들어왔으면 true (내용은 line에 들어 있다)
bool readLine() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      line[lineLength] = '\0';
      lineLength = 0;
      return true;
    }
    if (c != '\r' && lineLength < sizeof(line) - 1) {
      line[lineLength++] = c;
    }
  }
  return false;
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

void setup() {
  digitalWrite(PIN_BUZZER, BUZZER_OFF_LEVEL);  // 출력으로 바꾸기 전에 꺼 둔다 (켤 때 '삑' 소리 방지)
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_IR, INPUT);
  pinMode(PIN_RESET_BUTTON, INPUT_PULLUP);
  pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_LED_G, OUTPUT);
  pinMode(PIN_LED_B, OUTPUT);
  Serial.begin(9600);

  // 배선 확인: 빨강 → 초록 → 파랑 → '삑'
  setColor(true, false, false);
  delay(300);
  setColor(false, true, false);
  delay(300);
  setColor(false, false, true);
  delay(300);
  beep(1);

  // 켤 때 이미 닫혀 있는 문은 '닫힘'으로 치지 않는다
  door.stable = door.last = (digitalRead(PIN_IR) == DOOR_CLOSED_LEVEL);
  enter(IDLE);
}

void loop() {
  bool doorJustClosed = door.update(digitalRead(PIN_IR) == DOOR_CLOSED_LEVEL) && door.stable;
  bool buttonJustPressed = button.update(digitalRead(PIN_RESET_BUTTON) == LOW) && button.stable;
  bool gotLine = readLine();
  unsigned long elapsed = millis() - stateStart;

  switch (state) {
    case IDLE:
      if (doorJustClosed) {
        enter(SETTLING);
      } else if (buttonJustPressed) {
        Serial.println("RESET");  // 서버가 기준 사진을 다시 찍는다 (보관함을 비운 뒤 누른다)
        beep(1);
      }
      break;

    case SETTLING:  // 손이 빠지고 물건이 자리 잡을 때까지 기다린다
      if (!door.stable) {
        enter(IDLE);  // 문이 다시 열리면 취소하고, 닫히면 처음부터
      } else if (elapsed >= SETTLE_MS) {
        Serial.println("DROP");
        enter(WAITING);
      }
      break;

    case WAITING:  // 서버가 사진을 찍고 인식하는 중
      if (doorJustClosed) {
        pendingDrop = true;
      }
      if (gotLine && strcasecmp(line, "OK") == 0) {
        beep(2);
        backToIdle();
      } else if ((gotLine && strcasecmp(line, "FAIL") == 0) || elapsed >= RESPONSE_TIMEOUT_MS) {
        enter(SHOW_ERROR);
      }
      break;

    case SHOW_ERROR:
      if (doorJustClosed) {
        pendingDrop = true;
      }
      if (elapsed >= ERROR_SHOW_MS) {
        backToIdle();
      }
      break;
  }
}
