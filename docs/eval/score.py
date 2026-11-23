"""docs/eval의 평가 자료를 채점해서 보여 준다. (API 키 없이 돌아간다)

    python docs/eval/score.py

expected.csv와 lost_reports.json은 인식을 돌리기 전에 사진만 보고 적어 둔 것이고,
recognized_1~3.csv는 같은 사진 18장을 findbox_photos.py로 세 번 돌린 결과다.
"""
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))  # 저장소 맨 위의 findbox_match.py를 쓴다

from findbox_match import match, top_found


def _read(name: str) -> list[dict]:
    with open(HERE / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _choices(text: str) -> set[str]:
    return {value.strip() for value in text.split(",") if value.strip()}


def _grade(value: str, right: str, near: str) -> str:
    return "정답" if value in _choices(right) else "유사" if value in _choices(near) else "오답"


def score_recognition(expected: list[dict], recognized: list[dict], label: str) -> None:
    """한 번 돌린 결과를 정답표와 비교해 한 줄로 보여 주고, 틀린 것은 따로 적는다."""
    items, times = {}, {}
    for row in recognized:
        times[row["사진"]] = float(row["걸린 시간(초)"])
        if row["종류"]:
            items.setdefault(row["사진"], []).append(row)

    detected = category = color = private = total = 0
    wrong = []
    for row in expected:
        found = items.get(row["사진"], [])
        is_new = row["새로 들어온 물건"] != "(없음)"
        if len(found) != (1 if is_new else 0):
            wrong.append(f"{row['사진']}: 물건 {len(found)}개로 인식 (정답: {row['새로 들어온 물건']})")
            continue
        detected += 1
        if not is_new:
            continue
        total += 1
        item = found[0]
        grades = (_grade(item["종류"], row["종류 정답"], row["종류 유사"]), _grade(item["색상"], row["색상 정답"], row["색상 유사"]))
        category += grades[0] == "정답"
        color += grades[1] == "정답"
        private += item["비공개"] == row["비공개여야 함"]
        for what, grade, right in (("종류", grades[0], row["종류 정답"]), ("색상", grades[1], row["색상 정답"])):
            if grade != "정답":
                wrong.append(f"{row['새로 들어온 물건']}: {what} '{item[what]}' → {grade} (정답: {right})")
        if item["비공개"] != row["비공개여야 함"]:
            wrong.append(f"{row['새로 들어온 물건']}: 비공개 '{item['비공개']}' (정답: {row['비공개여야 함']})")

    seconds = list(times.values())
    print(f"  {label}  새 물건 판정 {detected}/{len(expected)}   종류 {category}/{total}   색상 {color}/{total}   "
          f"비공개 처리 {private}/{total}   한 장에 평균 {statistics.mean(seconds):.1f}초 (중앙값 {statistics.median(seconds):.1f}, 최대 {max(seconds):.1f})")
    for line in wrong:
        print(f"        ✗ {line}")


def score_matching(lost_reports: list[dict], recognized: list[dict]) -> None:
    """분실 신고마다 진짜 물건이 인식된 습득물 15개 가운데 몇 위인지 본다."""
    found_items = [{"photo": row["사진"], "name": row["이름"], "category": row["종류"], "color": row["색상"],
                    "features": row["특징"], "is_private": row["비공개"] == "예",
                    "registered_at": datetime.strptime(row["사진"][:15], "%Y%m%d_%H%M%S")}  # 사진 이름이 찍은 시각이다
                   for row in recognized if row["종류"]]
    first = top3 = 0
    for lost in lost_reports:
        scores = {found["photo"]: match(lost, found).score for found in found_items}
        mine = scores[lost["photo"]]
        rank = 1 + sum(score >= mine for photo, score in scores.items() if photo != lost["photo"])  # 동점은 불리하게 센다
        shown = [found["photo"] for found, _ in top_found(lost, found_items)]  # 웹에 보이는 상위 3개
        first += rank == 1
        top3 += lost["photo"] in shown
        other_photo, other = max(((p, s) for p, s in scores.items() if p != lost["photo"]), key=lambda pair: pair[1])
        other_name = next(found["name"] for found in found_items if found["photo"] == other_photo)
        print(f"  {rank}위 {mine:3d}점  {lost['name']} ({lost['color']})  ← 다음으로 높은 것: {other_name} {other}점"
              + ("" if lost["photo"] in shown else "   ✗ 상위 3개에 없음"))
    print(f"  → 진짜 물건이 1위: {first}/{len(lost_reports)}, 상위 3개 안: {top3}/{len(lost_reports)}")


def main():
    expected = _read("expected.csv")
    runs = [_read(f"recognized_{n}.csv") for n in (1, 2, 3)]
    print(f"인식: 사진 {len(expected)}장 (새 물건 {sum(r['새로 들어온 물건'] != '(없음)' for r in expected)}개 + 새 물건이 없어야 하는 사진 "
          f"{sum(r['새로 들어온 물건'] == '(없음)' for r in expected)}장)")
    for n, recognized in enumerate(runs, 1):
        score_recognition(expected, recognized, f"{n}회")

    print("\n매칭: 분실 신고 15건 × 1회 인식 결과의 습득물 15개")
    with open(HERE / "lost_reports.json", encoding="utf-8") as f:
        score_matching(json.load(f), runs[0])


if __name__ == "__main__":
    main()
