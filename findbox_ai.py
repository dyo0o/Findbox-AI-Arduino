import argparse
import csv
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import httpx
import numpy as np
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

MODEL = "gemini-3.8-flash"           # 모델 이름이 바뀌면 여기만 수정
THINKING = types.ThinkingLevel.LOW   # 새 물건 구분이 자주 틀리면 MEDIUM으로 (대신 느려짐)
MAX_SIDE = 1024                      # 사진 긴 변을 이 크기로 줄여서 보내고 저장 (속도·비용 절약)
TIMEOUT_MS = 20_000                  # 요청 1번당 최대 대기 시간 (Gemini 서버도 이 시간이 지나면 504로 포기함)
MAX_ATTEMPTS = 2                     # 서버 오류·사용량 초과·네트워크 오류일 때 시도 횟수

# 웹의 분실물 등록 폼·습득물 필터와 반드시 같은 목록이어야 매칭이 된다.
CATEGORIES = (
    "지갑", "무선 이어폰", "학생증/카드", "휴대폰", "텀블러/물병", "열쇠", "우산",
    "필통/필기구", "책/노트", "가방/파우치", "전자기기", "의류/잡화", "기타",
)
COLORS = ("흰색", "검정색", "파란색", "초록색", "빨간색", "갈색", "회색", "기타")
PRIVATE_CATEGORIES = {"학생증/카드"}  # 사진을 관리자에게만 공개하는 종류
MOSAIC_BLOCKS = 12                    # 비공개 물건 사진 모자이크 칸 수 (긴 변 기준, 작을수록 더 뭉개짐)

# 카메라 설정 (S 키로 연 설정 창에서 정한 값). 켤 때마다 이 값으로 고정한다.
CAMERA_FOCUS = 24            # 포커스 (자동 끔)
CAMERA_EXPOSURE = -8         # 노출 (자동 끔). -4 이상이면 1초에 30장을 못 찍는다
CAMERA_WHITE_BALANCE = 5000  # 화이트 밸런스 (자동 끔). 사진이 푸르면 올리고, 누러면 내린다
CAMERA_PROPS = {             # 나머지 값
    cv2.CAP_PROP_ZOOM: 100,
    cv2.CAP_PROP_BRIGHTNESS: 128,
    cv2.CAP_PROP_CONTRAST: 128,
    cv2.CAP_PROP_SATURATION: 128,
    cv2.CAP_PROP_SHARPNESS: 128,
    cv2.CAP_PROP_BACKLIGHT: 0,
    cv2.CAP_PROP_GAIN: 0,
}

SYSTEM_PROMPT = """너는 분실물 보관함의 습득물 인식 담당이다. 사진은 보관함 천장 카메라로 보관함 내부를 찍은 것이다.

규칙:
- 보관함 바닥·벽·조명·그림자는 물건이 아니다.
- category와 color는 정해진 목록에서만 고른다. 목록에 없는 물건은 category를 '기타'로 한다.
- color는 겉에 보이는 대표 색이다. 케이스나 커버가 씌워져 있으면 그 색을 고른다.
  목록에 맞는 색이 없으면 '기타'를 고르고 실제 색은 features에 적는다.
- name은 '에어팟 프로 케이스', '검정 반지갑'처럼 구체적으로 쓴다.
- features에는 브랜드, 스티커, 흠집, 키링처럼 주인을 찾는 데 도움이 되는 특징을 1~2문장으로 쓴다.
  사진에 보이는 것만 쓰고 추측하지 않는다.
- 이름, 학번, 전화번호, 카드번호 같은 개인정보는 name과 features에 절대 쓰지 않는다.
  개인정보가 보이는 물건은 has_personal_info를 true로 한다.
- box_2d는 물건을 감싸는 상자이며 [ymin, xmin, ymax, xmax] 순서로 0~1000으로 정규화한다."""

PROMPT_ALL = "보관함 안에 있는 물건을 모두 items에 담아라. 물건이 없으면 items는 빈 배열이다."

PROMPT_NEW = """사진 1은 물건을 넣기 전, 사진 2는 넣은 후의 보관함이다.
사진 2에 새로 생긴 물건만 items에 담아라. box_2d는 사진 2 기준이다.
- 원래 있던 물건이 밀리거나 움직인 것은 새 물건이 아니다.
- 사진 1에 있다가 사라진 물건은 무시한다.
- 새로 생긴 물건이 없으면 items는 빈 배열이다."""

log = logging.getLogger("findbox_ai")


class RecognitionError(Exception):
    """인식 실패. 메시지는 화면이나 로그에 그대로 보여 줘도 되는 문장이다."""


@dataclass
class FoundItem:
    category: str
    name: str
    color: str
    features: str
    is_private: bool  # True면 카드처럼 개인정보가 있는 물건 → 사진은 모자이크, 관리자에게만 공개
    photo: bytes      # 게시글 사진 (물건 부분만 잘라낸 JPEG, 비공개 물건은 글자를 못 읽게 모자이크)
    box: list[int]    # 보관함 사진 속 위치 [ymin, xmin, ymax, xmax], 0~1000

    def to_dict(self) -> dict:
        """DB에 넣을 값 (사진 제외)"""
        return {
            "category": self.category,
            "name": self.name,
            "color": self.color,
            "features": self.features,
            "is_private": self.is_private,
        }


# Gemini가 이 형식으로만 답하게 한다.
class _Item(BaseModel):
    category: Literal[CATEGORIES] = Field(description="물건 종류")
    name: str = Field(description="구체적인 물건 이름")
    color: Literal[COLORS] = Field(description="겉에 보이는 대표 색상")
    features: str = Field(description="주인을 찾는 데 도움이 되는 특징 1~2문장")
    has_personal_info: bool = Field(description="이름·학번·전화번호 같은 개인정보가 보이면 true")
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax], 0~1000으로 정규화한 좌표")


class _Answer(BaseModel):
    items: list[_Item]


def _inline_refs(schema: dict) -> dict:
    """pydantic이 만든 $ref를 실제 정의로 바꿔 넣는다. (Gemini에 단순한 스키마를 보내기 위해)"""
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(value) for value in node]
        return node

    return resolve(schema)


_SCHEMA = _inline_refs(_Answer.model_json_schema())
_LONG_NUMBER = re.compile(r"(?:\d[\s-]?){6,}\d")  # 7자리 이상 숫자 (학번·전화번호·카드번호)
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RecognitionError("GEMINI_API_KEY 환경변수가 없습니다. API 키를 먼저 설정하세요.")
        _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client


def load_image(source: str | Path | bytes) -> np.ndarray:
    """사진 파일 경로나 JPEG/PNG bytes를 OpenCV 이미지로 읽는다. (한글 경로 가능)"""
    if isinstance(source, bytes):
        data = np.frombuffer(source, np.uint8)
    elif Path(source).is_file():
        data = np.fromfile(source, np.uint8)
    else:
        raise ValueError(f"사진 파일이 없습니다: {source}")
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("사진을 읽을 수 없습니다.")
    return frame


def _to_jpeg(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _box_area(frame: np.ndarray, box: list[int], margin: float = 0.1) -> tuple[int, int, int, int] | None:
    """box_2d(0~1000)를 여백을 둔 픽셀 범위 (top, bottom, left, right)로 바꾼다. 좌표가 이상하면 None."""
    if len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = (min(max(v, 0), 1000) for v in box)
    dy, dx = (ymax - ymin) * margin, (xmax - xmin) * margin
    h, w = frame.shape[:2]
    top, bottom = max(0, int((ymin - dy) * h / 1000)), min(h, int((ymax + dy) * h / 1000))
    left, right = max(0, int((xmin - dx) * w / 1000)), min(w, int((xmax + dx) * w / 1000))
    if bottom - top < 2 or right - left < 2:
        return None
    return top, bottom, left, right


def _crop(frame: np.ndarray, box: list[int]) -> np.ndarray:
    """물건 부분을 여백을 두고 잘라낸다. 좌표가 이상하면 전체 사진을 쓴다."""
    area = _box_area(frame, box)
    if area is None:
        return frame
    top, bottom, left, right = area
    return frame[top:bottom, left:right]


def _mosaic(image: np.ndarray) -> np.ndarray:
    """글자를 못 읽게 모자이크한다. (색과 대략적인 모양은 남는다)"""
    h, w = image.shape[:2]
    scale = MOSAIC_BLOCKS / max(h, w)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def hide_private(frame: np.ndarray, items: list["FoundItem"]) -> np.ndarray:
    """보관함 전체 사진에서 비공개 물건(카드 등) 부분을 모자이크한 사본을 돌려준다.

    위치를 알 수 없는 비공개 물건이 있으면 안전하게 사진 전체를 모자이크한다.
    """
    frame = frame.copy()
    for item in items:
        if not item.is_private:
            continue
        area = _box_area(frame, item.box)
        if area is None:
            return _mosaic(frame)
        top, bottom, left, right = area
        frame[top:bottom, left:right] = _mosaic(frame[top:bottom, left:right])
    return frame


def _hide_numbers(text: str) -> str:
    """Gemini가 실수로 적은 학번·전화번호 같은 긴 숫자를 가린다."""
    return _LONG_NUMBER.sub("***", text).strip()


def _request(contents: list) -> _Answer:
    """Gemini에 요청해서 답을 받는다. 일시적인 오류는 다시 시도한다."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    client = _get_client()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=contents, config=config)
            return _Answer.model_validate_json(response.text or "")
        except errors.ClientError as e:
            if e.code != 429:  # 키·모델 이름·요청 형식 오류는 다시 해도 같다
                raise RecognitionError(f"요청 오류 ({e.code}): {e.message}") from e
            error = "사용량 한도를 넘었습니다 (429)."
        except errors.ServerError as e:
            error = f"Gemini 서버 오류 ({e.code}): {e.message}"
        except httpx.TimeoutException:
            error = f"{TIMEOUT_MS // 1000}초 안에 응답이 오지 않았습니다."
        except httpx.TransportError:
            error = "Gemini 서버에 연결할 수 없습니다. 인터넷 연결을 확인하세요."
        except ValidationError:
            error = "Gemini 응답이 정해진 형식이 아닙니다."
        log.info("인식 실패, 다시 시도 (%d/%d): %s", attempt, MAX_ATTEMPTS, error)  # 화면에는 안 보이고 서버 로그용
        if attempt < MAX_ATTEMPTS:
            time.sleep(attempt)
    raise RecognitionError(error)


def analyze(after: np.ndarray, before: np.ndarray | None = None) -> list[FoundItem]:
    """보관함 사진을 분석해서 습득물 목록을 돌려준다.

    before(넣기 전 사진)가 있으면 새로 들어온 물건만, 없으면 사진 속 물건을 모두 돌려준다.
    물건이 없으면 빈 리스트. 인식에 실패하면 RecognitionError.
    """
    def image(frame):
        return types.Part.from_bytes(data=_to_jpeg(frame), mime_type="image/jpeg")

    if before is None:
        contents = [image(after), PROMPT_ALL]
    else:
        contents = ["사진 1 (넣기 전)", image(before), "사진 2 (넣은 후)", image(after), PROMPT_NEW]

    start = time.perf_counter()
    answer = _request(contents)
    log.info("인식 완료: %d개, %.1f초", len(answer.items), time.perf_counter() - start)

    found = []
    for item in answer.items:
        is_private = item.category in PRIVATE_CATEGORIES or item.has_personal_info
        photo = _crop(after, item.box_2d)
        if is_private:  # 원본은 어디에도 남기지 않고, 처음부터 모자이크된 사진만 넘긴다
            photo = _mosaic(photo)
        found.append(FoundItem(
            category=item.category,
            name=_hide_numbers(item.name),
            color=item.color,
            features=_hide_numbers(item.features),
            is_private=is_private,
            photo=_to_jpeg(photo),
            box=item.box_2d,
        ))
    return found


class Camera:
    """보관함 웹캠. 서버가 켜져 있는 동안 계속 열어 둔다."""

    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"웹캠({index}번)을 열 수 없습니다. 카메라 번호를 확인하세요.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 고해상도에서 1초에 30장 찍으려면 MJPG 형식이 필요하다 (아니면 5장 정도로 떨어짐).
        # Windows(DSHOW)에서는 해상도를 바꿀 때 형식이 초기화되므로 반드시 해상도 다음에 설정한다.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._fix_settings()
        for _ in range(5):  # 켠 직후 프레임은 노출이 안정되지 않아서 버림
            self.cap.read()
        start = time.perf_counter()
        for _ in range(10):  # 실제로 1초에 몇 장 찍히는지 잰다
            self.cap.read()
        self.fps = 10 / (time.perf_counter() - start)
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)) if code else "알 수 없음"
        if self.fps < 10:
            log.warning("카메라가 1초에 %.0f장만 찍어서 화면이 끊겨 보일 수 있습니다. (형식 %s, 조명이 어둡거나 MJPG가 아닐 때)",
                        self.fps, self.codec)

    def _fix_settings(self):
        """설정 창에서 정한 값으로 카메라를 고정한다. (카메라를 다시 꽂으면 초기화되므로 켤 때마다 적용)"""
        cap = self.cap
        for prop, value in CAMERA_PROPS.items():
            cap.set(prop, value)
        # 자동 끄기 → 값 넣기 → 다시 자동 끄기 순서로 해야 그 값으로 수동 고정된다
        for auto, prop, value in ((cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_FOCUS, CAMERA_FOCUS),
                                  (cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)):
            cap.set(auto, 0)
            cap.set(prop, value)
            cap.set(auto, 0)
        # 화이트 밸런스는 자동을 끌 때 기본값으로 돌아가므로, 끈 다음에 값을 넣는다
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, CAMERA_WHITE_BALANCE)
        applied = {"포커스": (cv2.CAP_PROP_FOCUS, CAMERA_FOCUS), "노출": (cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE),
                   "화이트 밸런스": (cv2.CAP_PROP_WB_TEMPERATURE, CAMERA_WHITE_BALANCE)}
        wrong = [f"{name} {cap.get(prop):.0f} (설정값 {value})" for name, (prop, value) in applied.items()
                 if abs(cap.get(prop) - value) > 1]
        if wrong:
            log.warning("카메라 설정이 적용되지 않았습니다: %s", ", ".join(wrong))

    def open_settings(self):
        """카메라 설정 창을 연다 (Windows). 자동 노출·저조도 보정·초점을 여기서 바꿀 수 있다."""
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def read(self) -> np.ndarray:
        """다음 프레임을 읽는다. (미리보기처럼 계속 읽을 때)"""
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("웹캠에서 사진을 찍지 못했습니다.")
        return frame

    def capture(self) -> np.ndarray:
        """지금 장면을 찍는다. (가끔 한 장씩 찍을 때)"""
        for _ in range(5):  # 버퍼에 남아 있는 예전 프레임을 버림
            self.cap.grab()
        return self.read()

    def close(self):
        self.cap.release()


class _Saver:
    """결과 사진을 실행할 때마다 새 폴더(results/날짜-시간)에 순서대로 번호를 붙여 저장한다.

    예) 01_기준.jpg → 02_전체.jpg, 02_물건1.jpg → 03_기준.jpg(B를 눌렀을 때) → 04_전체.jpg ...
    """

    def __init__(self, out_dir: Path):
        self.dir = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.n = 0

    def next(self):
        self.n += 1

    def save(self, name: str, jpeg: bytes) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self.n:02d}_{name}.jpg"
        path.write_bytes(jpeg)
        return path

    def record(self, number: int, item: FoundItem):
        """인식 결과를 인식결과.csv에 한 줄씩 쌓는다."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "인식결과.csv"
        is_new = not path.exists()
        with open(path, "a", encoding="utf-8-sig", newline="") as f:  # 엑셀에서 한글이 안 깨지게
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["번호", "종류", "이름", "색상", "특징", "비공개", "등록 시간"])
            writer.writerow([f"{self.dir.name[-6:]}-{self.n:02d}-{number}",  # 실행 시각-사진 번호-물건 번호
                             item.category, item.name, item.color, item.features,
                             "예" if item.is_private else "아니오", datetime.now().strftime("%Y-%m-%d %H:%M")])


def _show(items: list[FoundItem], elapsed: float, after: np.ndarray, saver: _Saver, hidden: list[FoundItem]):
    saver.next()
    saver.save("전체", _to_jpeg(hide_private(after, hidden)))
    if not items:
        print(f"새 물건 없음 ({elapsed:.1f}초)")
        return
    print(f"물건 {len(items)}개 인식 ({elapsed:.1f}초)")
    for i, item in enumerate(items, 1):
        saver.save(f"물건{i}", item.photo)
        saver.record(i, item)
        print(f"[{i}] {item.category} · {item.color} · {item.name}" + (" [비공개]" if item.is_private else ""))
        print(f"    특징: {item.features}")


def _save_base(frame: np.ndarray, saver: _Saver, hidden: list[FoundItem]):
    """기준 사진도 저장해 둔다. (새 물건이 왜 안 잡혔는지 나중에 비교해 볼 수 있게)"""
    saver.next()
    saver.save("기준", _to_jpeg(hide_private(frame, hidden)))


WINDOW = "FindBox Camera"  # OpenCV 창에는 한글이 깨져서 나오므로 창 제목과 화면 글자는 영어로 쓴다
RESULT_SECONDS = 5         # 인식 결과(초록 상자)를 화면에 보여 주는 시간


def _put_text(view: np.ndarray, text: str, org: tuple[int, int], color=(255, 255, 255)):
    # 글자는 한 번만 그린다. OpenCV 5는 굵기에 따라 글자 폭이 달라져서, 굵기를 바꿔 겹쳐 그리면 어긋난다.
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def _draw(frame: np.ndarray, items: list[FoundItem], status: str, fps: float) -> np.ndarray:
    """미리보기 화면에 마지막 인식 결과(초록 상자)와 안내 문구를 그린다."""
    h, w = frame.shape[:2]
    scale = min(1.0, 960 / w)  # 화면에 맞게 줄여서 보여 준다
    view = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame.copy()
    vh, vw = view.shape[:2]
    for i, item in enumerate(items, 1):
        if len(item.box) != 4:
            continue
        ymin, xmin, ymax, xmax = (min(max(v, 0), 1000) for v in item.box)
        x, y = int(xmin * vw / 1000), int(ymin * vh / 1000)
        cv2.rectangle(view, (x, y), (int(xmax * vw / 1000), int(ymax * vh / 1000)), (0, 220, 0), 2)
        label = f"[{i}]"  # 터미널에 나온 번호와 같다
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(view, (x, y), (x + tw + 8, y + th + 10), (0, 220, 0), -1)
        _put_text(view, label, (x + 4, y + th + 5), (0, 0, 0))
    view[:62] = (view[:62] * 0.35).astype(np.uint8)  # 안내 문구가 잘 보이게 위쪽을 어둡게
    _put_text(view, f"SPACE: analyze   B: new baseline   S: camera settings   Q: quit   ({fps:.0f} FPS)", (10, 25))
    _put_text(view, status, (10, 52))
    return view


def _camera_loop(index: int, saver: _Saver):
    camera = Camera(index)
    try:
        before = camera.capture()
        hidden: list[FoundItem] = []  # 이번 실행에서 나온 비공개 물건 (저장 사진에서 계속 가린다)
        _save_base(before, saver, hidden)
        # 키 목록은 카메라 창 위쪽에 나오므로 여기서는 꼭 알아야 할 것만 알린다
        print(f"카메라 창을 클릭하고 Space로 인식합니다. 지금 있는 물건은 인식하지 않습니다. (결과 사진: {saver.dir})")
        items, status, shown_at = [], "Ready", 0.0
        job, result = None, {}
        fps, frames, fps_start = camera.fps, 0, time.perf_counter()
        while True:
            frame = camera.read()
            frames += 1
            if time.perf_counter() - fps_start >= 1:  # 1초마다 화면 FPS 갱신
                fps = frames / (time.perf_counter() - fps_start)
                frames, fps_start = 0, time.perf_counter()

            if job is not None and not job.is_alive():  # 인식이 끝났으면 결과 반영
                elapsed = time.perf_counter() - start
                if "items" in result:
                    items = result["items"]
                    if items:  # 새 물건을 찾았을 때만 기준 사진을 바꾼다 (놓친 물건은 다음 인식 때 다시 비교)
                        before = after
                    status = f"{len(items)} new item(s) ({elapsed:.1f}s)"
                    hidden += [item for item in items if item.is_private]
                    _show(items, elapsed, after, saver, hidden)
                else:
                    items, status = [], "Failed - see terminal"
                    print(f"인식 실패: {result.get('error', '알 수 없는 오류')} → 다시 Space를 누르세요")
                job, shown_at = None, time.perf_counter()

            if job is None and time.perf_counter() - shown_at > RESULT_SECONDS:  # 결과는 잠깐만 보여 준다
                items, status = [], "Ready"

            cv2.imshow(WINDOW, _draw(frame, items, "Analyzing..." if job else status, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                camera.open_settings()
            if key in (ord("b"), ord("B")) and job is None:  # 지금 화면을 새 기준 사진으로
                before = frame
                _save_base(before, saver, hidden)
                items, status, shown_at = [], "New baseline", time.perf_counter()  # 카메라 창에 표시된다
            if key in (ord(" "), 13) and job is None:  # Space 또는 Enter
                after, start, result = frame, time.perf_counter(), {}

                def work(after=after, before=before, result=result):
                    try:
                        result["items"] = analyze(after, before)
                    except RecognitionError as e:
                        result["error"] = e

                # 인식하는 몇 초 동안에도 화면이 멈추지 않게 따로 돌린다
                job = threading.Thread(target=work, daemon=True)
                job.start()
    finally:
        camera.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="FindBox 습득물 인식 (시연 전 카메라·인식 테스트)")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호 (기본 0번)")
    parser.add_argument("--out", default="results", help="결과 사진을 저장할 폴더 (기본 results, 실행할 때마다 안에 새 폴더)")
    args = parser.parse_args()
    # 터미널에는 이 프로그램의 경고(카메라 문제 등)만 보이게 하고, 다른 라이브러리의 경고는 숨긴다
    logging.basicConfig(format="※ %(message)s", level=logging.ERROR)
    log.setLevel(logging.WARNING)
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV의 [ WARN ] 줄

    try:
        if not os.environ.get("GEMINI_API_KEY"):
            print("GEMINI_API_KEY가 없어서 카메라 화면만 볼 수 있습니다. 인식하려면 API 키를 설정하세요.")
        _camera_loop(args.camera, _Saver(Path(args.out)))
    except (RecognitionError, RuntimeError, ValueError, OSError) as e:
        sys.exit(f"오류: {e}")
    except KeyboardInterrupt:  # Ctrl+C로 꺼도 오류 메시지 없이 끝낸다
        pass


if __name__ == "__main__":
    main()
