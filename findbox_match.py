import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

# 항목별 비중 (합이 100이 아니어도 된다). 시연은 보관함이 하나라 장소는 0
WEIGHTS = {"종류": 30, "색상": 25, "특징": 35, "시간": 10, "장소": 0}
CATEGORY_MISMATCH_CAP = 25               # 종류가 확실히 다르면 최대 이 점수까지만 (MIN_SCORE보다 낮게 → 결과에서 빠짐)
MIN_SCORE = 30                           # 이 점수보다 낮으면 매칭 결과에서 뺀다
TIME_TOLERANCE = timedelta(minutes=30)   # 분실 시간은 대충 적으니, 이만큼 먼저 등록된 건 봐준다
FEATURE_FULL = 0.8                       # 분실물 특징의 이 비율 이상이 습득물 설명에 있으면 만점
LEVELS = ((85, "높은 일치 가능성"), (60, "일치 가능성 있음"), (0, "낮은 일치 가능성"))
PRIVATE_NOTICE = "개인정보 보호 물품이라 자세한 정보는 공개하지 않습니다. 신분증을 가지고 관리자에게 방문하세요."

# 서로 헷갈리기 쉬운 종류·색상 (0~1점)
SIMILAR_CATEGORIES = {
    frozenset({"무선 이어폰", "전자기기"}): 0.5,
    frozenset({"휴대폰", "전자기기"}): 0.5,
    frozenset({"지갑", "학생증/카드"}): 0.5,       # 카드지갑, 지갑에 든 카드
    frozenset({"지갑", "가방/파우치"}): 0.5,
    frozenset({"필통/필기구", "가방/파우치"}): 0.5,
}
SIMILAR_COLORS = {
    frozenset({"흰색", "회색"}): 0.5,
    frozenset({"검정색", "회색"}): 0.5,
    frozenset({"검정색", "갈색"}): 0.5,            # 어두운 갈색 가죽
    frozenset({"빨간색", "갈색"}): 0.3,
    frozenset({"파란색", "초록색"}): 0.3,
}

# 특징을 비교할 때 같은 단어로 볼 것
SYNONYMS = {
    "airpods": "에어팟", "airpod": "에어팟", "iphone": "아이폰", "galaxy": "갤럭시", "buds": "버즈",
    "apple": "애플", "samsung": "삼성", "보틀": "물병", "물통": "물병", "키링": "열쇠고리", "키홀더": "열쇠고리",
}
# 특징을 비교할 때 빼는 단어 (색상은 색상 항목에서 따로 비교하므로 뺀다)
_COLOR_WORDS = {
    "흰색", "하얀색", "하얀", "흰", "화이트", "검정", "검정색", "검은색", "검은", "까만", "블랙",
    "파란", "파란색", "파랑", "남색", "하늘색", "네이비", "블루", "초록", "초록색", "녹색", "연두색", "그린",
    "빨간", "빨간색", "빨강", "붉은", "레드", "갈색", "브라운", "회색", "그레이",
    "노란", "노란색", "노랑", "분홍", "분홍색", "핑크", "보라", "보라색", "주황", "주황색", "베이지", "색", "색상",
}
_STOPWORDS = {
    "있다", "있음", "있는", "있고", "있어", "있어요", "있으며", "있습니다", "있던", "없다", "없음", "없는", "없어요",
    "붙어", "붙은", "달린", "달려", "약간", "조금", "그리고", "그냥", "같은", "모양", "부분", "쪽", "물건", "제품",
    "것", "정도", "특별한", "표시", "보입니다", "입니다", "되어", "케이스",
}
_PARTICLES = ("입니다", "에서", "으로", "에게", "부터", "까지", "이며", "이고", "이다",
              "은", "는", "을", "를", "에", "의", "가", "도", "와", "과", "로")  # '이'는 곰돌이 같은 단어가 망가져서 뺐다
_NON_WORD = re.compile(r"[^0-9a-z가-힣]+")


@dataclass
class Reason:
    item: str    # 종류 / 색상 / 특징 / 시간 / 장소
    result: str  # 일치 / 유사 / 불일치 / 확인 필요 / 정보 없음
    text: str    # 화면에 보여 줄 설명


@dataclass
class MatchResult:
    score: int              # 0~100
    level: str              # 높은 일치 가능성 / 일치 가능성 있음 / 낮은 일치 가능성
    summary: str            # 한 문장 매칭 이유
    reasons: list[Reason]   # 항목별 매칭 이유
    excluded: bool = False  # True면 같은 물건일 수 없음 (분실 시간보다 먼저 들어온 습득물)
    private: bool = False   # 습득물이 비공개 물건(카드 등)인지

    def to_dict(self, admin: bool = False) -> dict:
        """웹에 보낼 값. 비공개 습득물은 관리자가 아니면 매칭률만 준다."""
        if self.private and not admin:
            return {"score": self.score, "level": self.level, "excluded": self.excluded,
                    "private": True, "notice": PRIVATE_NOTICE}
        return {"score": self.score, "level": self.level, "excluded": self.excluded,
                "private": self.private, "summary": self.summary,
                "reasons": [asdict(reason) for reason in self.reasons]}


def _josa(word: str, with_batchim: str, without_batchim: str) -> str:
    """받침에 따라 조사를 고른다. (예: 색상이 / 종류가)"""
    last = word[-1] if word else ""
    if "가" <= last <= "힣":
        return with_batchim if (ord(last) - 0xAC00) % 28 else without_batchim
    return with_batchim


def _to_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _duration(gap: timedelta) -> str:
    minutes = int(gap.total_seconds() // 60)
    if minutes < 60:
        return f"{max(minutes, 1)}분"
    if minutes < 48 * 60:
        return f"{minutes // 60}시간"
    return f"{minutes // (24 * 60)}일"


def _words(text: str) -> list[str]:
    """특징 비교용 단어 목록. 조사·색상·흔한 말을 빼고 같은 뜻 단어는 하나로 맞춘다."""
    words = []
    for word in _NON_WORD.sub(" ", (text or "").lower()).split():
        for particle in _PARTICLES:
            if word.endswith(particle) and len(word) - len(particle) >= 2:
                word = word[:-len(particle)]
                break
        word = SYNONYMS.get(word, word)
        if word not in _STOPWORDS and word not in _COLOR_WORDS:
            words.append(word)
    return words


def _pieces(words: list[str]) -> set[str]:
    """두 글자씩 자른 조각. '스티커가'와 '스티커'처럼 형태가 조금 달라도 겹치게 하려고 쓴다."""
    pieces = set()
    for word in words:
        pieces |= {word} if len(word) == 1 else {word[i:i + 2] for i in range(len(word) - 1)}
    return pieces


def _compare_category(lost: str, found: str):
    if not lost or not found:
        return None, "정보 없음", "종류 정보가 없어 비교하지 않았습니다."
    if lost == found == "기타":
        return 0.5, "유사", "둘 다 '기타'라서 이름과 특징으로 비교합니다."
    if lost == found:
        return 1.0, "일치", f"둘 다 '{lost}'입니다."
    if "기타" in (lost, found):
        return 0.3, "확인 필요", "한쪽이 '기타'라서 종류만으로는 판단하기 어렵습니다."
    similar = SIMILAR_CATEGORIES.get(frozenset((lost, found)))
    if similar:
        return similar, "유사", f"'{lost}'{_josa(lost, '과', '와')} '{found}'{_josa(found, '은', '는')} 헷갈리기 쉬운 종류입니다."
    return 0.0, "불일치", f"종류가 다릅니다. (분실물 '{lost}', 습득물 '{found}')"


def _compare_color(lost: str, found: str):
    if not lost or not found:
        return None, "정보 없음", "색상 정보가 없어 비교하지 않았습니다."
    if lost == found == "기타":
        return 0.6, "유사", "둘 다 목록에 없는 색('기타')입니다."
    if lost == found:
        return 1.0, "일치", f"둘 다 '{lost}'입니다."
    if "기타" in (lost, found):
        return 0.3, "확인 필요", "한쪽 색상이 '기타'라서 정확히 비교하기 어렵습니다."
    similar = SIMILAR_COLORS.get(frozenset((lost, found)))
    if similar:
        return similar, "유사", f"'{lost}'{_josa(lost, '과', '와')} '{found}'{_josa(found, '은', '는')} 조명에 따라 비슷해 보일 수 있습니다."
    return 0.0, "불일치", f"색상이 다릅니다. (분실물 '{lost}', 습득물 '{found}')"


def _compare_features(lost: dict, found: dict):
    lost_words = _words(f"{lost.get('features', '')} {lost.get('name', '')}")  # 겹친 단어는 특징부터 보여 준다
    found_words = _words(f"{found.get('name', '')} {found.get('features', '')}")
    lost_pieces = _pieces(lost_words)
    if not lost_pieces:
        return None, "정보 없음", "분실물에 적힌 이름·특징이 없어 비교하지 않았습니다."
    contained = len(lost_pieces & _pieces(found_words)) / len(lost_pieces)
    score = min(1.0, contained / FEATURE_FULL)
    result = "일치" if score >= 0.99 else "유사" if score >= 0.3 else "불일치"

    found_text = " ".join(found_words)
    common = list(dict.fromkeys(word for word in lost_words if len(word) >= 2 and word in found_text))[:3]
    if common:
        listed = ", ".join(f"'{word}'" for word in common)
        return score, result, f"{listed}{_josa(common[-1], '이', '가')} 겹칩니다."
    if result == "유사":
        return score, result, "비슷한 표현이 있습니다."
    return score, result, "겹치는 특징이 없습니다."


def _compare_time(lost: dict, found: dict):
    """(점수, 결과, 설명, 같은 물건일 수 없는지)"""
    lost_at, found_at = _to_datetime(lost.get("lost_at")), _to_datetime(found.get("registered_at"))
    if lost_at is None or found_at is None:
        return None, "정보 없음", "분실 시간이 없어 비교하지 않았습니다.", False
    gap = found_at - lost_at
    if gap < -TIME_TOLERANCE:
        return 0.0, "불일치", "분실 시간보다 먼저 보관함에 들어온 물건이라 같은 물건일 수 없습니다.", True
    if gap <= timedelta(0):
        return 1.0, "일치", "분실 시간과 거의 같은 때에 등록되었습니다.", False
    for limit, score, result in ((timedelta(hours=3), 1.0, "일치"), (timedelta(days=1), 0.7, "유사"),
                                 (timedelta(days=3), 0.4, "유사"), (timedelta(days=7), 0.2, "불일치")):
        if gap <= limit:
            break
    else:
        score, result = 0.1, "불일치"
    return score, result, f"분실하고 {_duration(gap)} 뒤에 등록되었습니다.", False


def _compare_place(lost: dict, found: dict):
    place, location = lost.get("place") or "", (found.get("location") or "").split()
    if not place or not location:
        return None, "정보 없음", "분실 장소나 보관 위치가 없어 비교하지 않았습니다."
    building = location[0]  # 보관 위치의 첫 단어를 건물 이름으로 본다 (예: 신텍스 FindBox 부스 → 신텍스)
    if building in place:
        return 1.0, "일치", f"분실 장소와 같은 건물({building})의 보관함입니다."
    return 0.0, "불일치", f"분실 장소와 다른 건물({building})의 보관함입니다."


def _summary(reasons: list[Reason], excluded: bool) -> str:
    if excluded:
        return "분실 시간보다 먼저 들어온 습득물이라 같은 물건일 수 없습니다."
    if any(r.item == "종류" and r.result == "불일치" for r in reasons):
        return "종류가 달라 같은 물건일 가능성이 낮습니다."
    names = {"시간": "시간대"}
    same = [names.get(r.item, r.item) for r in reasons if r.result == "일치"]
    near = [names.get(r.item, r.item) for r in reasons if r.result == "유사"]
    parts = []
    if same:
        parts.append(f"{'·'.join(same)}{_josa(same[-1], '이', '가')} 일치")
    if near:
        parts.append(f"{'·'.join(near)}{_josa(near[-1], '이', '가')} 비슷")
    return "하고, ".join(parts) + "합니다." if parts else "일치하는 항목이 거의 없습니다."


def match(lost: dict, found: dict) -> MatchResult:
    """분실물 1개와 습득물 1개를 비교한다."""
    time_score, time_result, time_text, excluded = _compare_time(lost, found)
    compared = {
        "종류": _compare_category(lost.get("category"), found.get("category")),
        "색상": _compare_color(lost.get("color"), found.get("color")),
        "특징": _compare_features(lost, found),
        "시간": (time_score, time_result, time_text),
        "장소": _compare_place(lost, found),
    }

    reasons, total, weight_sum = [], 0.0, 0
    for item, (score, result, text) in compared.items():
        weight = WEIGHTS.get(item, 0)
        if weight <= 0:
            continue  # 비중이 0인 항목은 점수에도, 이유에도 넣지 않는다
        reasons.append(Reason(item, result, text))
        if score is not None:  # 정보가 없는 항목은 빼고 나머지 비중으로 계산한다
            total += weight * score
            weight_sum += weight

    score = round(total / weight_sum * 100) if weight_sum else 0
    if compared["종류"][0] == 0.0:
        score = min(score, CATEGORY_MISMATCH_CAP)
    if excluded:
        score = 0
    level = next(name for limit, name in LEVELS if score >= limit)
    return MatchResult(score, level, _summary(reasons, excluded), reasons,
                       excluded=excluded, private=bool(found.get("is_private")))


def top_found(lost: dict, found_items: list[dict], n: int = 3, min_score: int = MIN_SCORE):
    """분실물 하나에 맞는 습득물을 점수 높은 순으로 n개까지 고른다. [(found, result), ...]"""
    results = [(found, match(lost, found)) for found in found_items]
    results = [(found, r) for found, r in results if not r.excluded and r.score >= min_score]
    results.sort(key=lambda pair: pair[1].score, reverse=True)
    return results[:n]


def top_lost(found: dict, lost_items: list[dict], n: int | None = None, min_score: int = MIN_SCORE):
    """새로 들어온 습득물 하나와 맞는 분실물을 점수 높은 순으로 고른다. [(lost, result), ...]"""
    results = [(lost, match(lost, found)) for lost in lost_items]
    results = [(lost, r) for lost, r in results if not r.excluded and r.score >= min_score]
    results.sort(key=lambda pair: pair[1].score, reverse=True)
    return results if n is None else results[:n]


def _demo():
    """예시 데이터로 매칭 결과를 보여 준다."""
    today = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
    lost = {"name": "에어팟 프로", "category": "무선 이어폰", "color": "흰색",
            "features": "케이스에 곰돌이 스티커가 붙어 있어요", "place": "신텍스 3층 강의실", "lost_at": today}
    box = "신텍스 FindBox 부스"
    found_items = [
        {"name": "에어팟 프로 케이스", "category": "무선 이어폰", "color": "흰색",
         "features": "앞면에 곰돌이 스티커가 붙어 있다.", "registered_at": today + timedelta(minutes=65)},
        {"name": "에어팟 프로 케이스", "category": "무선 이어폰", "color": "흰색",
         "features": "특별한 표시가 없는 흰색 케이스", "registered_at": today + timedelta(minutes=80)},
        {"name": "갤럭시 버즈 케이스", "category": "무선 이어폰", "color": "검정색",
         "features": "모서리에 흠집이 있다.", "registered_at": today + timedelta(minutes=90)},
        {"name": "흰색 텀블러", "category": "텀블러/물병", "color": "흰색",
         "features": "곰돌이 그림이 있는 스테인리스 텀블러", "registered_at": today + timedelta(minutes=100)},
        {"name": "에어팟 프로 케이스", "category": "무선 이어폰", "color": "흰색",
         "features": "곰돌이 스티커", "registered_at": today - timedelta(hours=2)},
    ]
    for found in found_items:
        found["location"] = box

    def label(found):
        return f"{found['name']} ({found['color']}, {found['registered_at']:%H:%M} 등록)"

    print(f"분실물: {lost['name']} ({lost['color']}) / {lost['features']} / {lost['lost_at']:%H:%M} 분실\n")

    # 매칭 결과 화면에 뜨는 순서대로: 항목별 결과 + 겹친 특징
    ranked = top_found(lost, found_items, n=len(found_items))
    for rank, (found, result) in enumerate(ranked, 1):
        items = " · ".join(f"{reason.item} {reason.result}" for reason in result.reasons)
        feature = next((reason.text for reason in result.reasons
                        if reason.item == "특징" and reason.result != "정보 없음"), "")
        print(f"{rank}위 {result.score:3d}점 {result.level}  {label(found)}")
        print(f"     {items}" + (f" — {feature}" if feature else ""))

    # 결과에서 빠진 습득물과 그 이유
    shown = {id(found) for found, _ in ranked}
    for found in found_items:
        if id(found) in shown:
            continue
        result = match(lost, found)
        different_kind = any(r.item == "종류" and r.result == "불일치" for r in result.reasons)
        why = result.summary if result.excluded or different_kind else f"{result.score}점 (기준 {MIN_SCORE}점 미만)"
        print(f"제외  {label(found)} — {why}")


if __name__ == "__main__":
    _demo()
