"""사진 파일로 습득물 인식을 해 본다. (웹캠·보관함 없이 findbox_ai.py의 인식을 확인하는 도구)

    python findbox_photos.py 사진.jpg     사진에 보이는 물건을 모두 인식한다
    python findbox_photos.py 사진폴더     폴더 안 사진을 이름순으로 읽고, 바로 앞 사진과 비교해 새로 들어온 물건만 인식한다
                                          (첫 사진은 기준 사진이라 인식하지 않는다)
"""
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from findbox_ai import RecognitionError, analyze, load_image

PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
MAX_TRIES = 3    # 사진 한 장을 인식해 보는 횟수 (사용량 한도·서버 오류처럼 잠시 뒤 다시 하면 되는 오류일 때)
RETRY_WAIT = 20  # 다시 해 보기 전에 기다리는 시간 (초)


def _in_order(path: Path) -> list:
    """이름순 정렬 기준. 숫자는 숫자로 비교해서 2.jpg가 10.jpg보다 앞에 온다."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _pairs(paths: list[Path]) -> list[tuple[Path, Path | None]]:
    """(인식할 사진, 비교할 앞 사진) 목록. 사진 파일을 직접 주면 앞 사진 없이 보이는 물건을 모두 인식한다."""
    pairs = []
    for path in paths:
        if path.is_dir():
            photos = sorted((p for p in path.iterdir() if p.suffix.lower() in PHOTO_TYPES), key=_in_order)
            pairs += zip(photos[1:], photos)
        else:
            pairs.append((path, None))
    return pairs


def _recognize(after, before):
    """(인식한 물건, 걸린 시간). 잠시 뒤 다시 하면 되는 오류는 기다렸다가 다시 해 보고, 끝내 안 되면 (None, 0)."""
    for attempt in range(1, MAX_TRIES + 1):
        start = time.perf_counter()
        try:
            return analyze(after, before), time.perf_counter() - start
        except RecognitionError as e:
            if not e.retry:  # API 키·모델 이름 오류는 다음 사진도 똑같이 실패한다
                sys.exit(f"오류: {e} → 고친 뒤 다시 실행하세요")
            again = attempt < MAX_TRIES
            print(f"    인식 실패: {e}" + (f" → {RETRY_WAIT}초 뒤 다시 시도" if again else ""))
            if again:
                time.sleep(RETRY_WAIT)
    return None, 0.0


def main():
    parser = argparse.ArgumentParser(description="FindBox 습득물 인식을 사진 파일로 해 보기 (웹캠 없이)")
    parser.add_argument("paths", nargs="+", type=Path, metavar="사진_또는_폴더",
                        help="사진 파일이면 보이는 물건을 모두, 폴더면 이름순으로 앞 사진과 비교해 새로 들어온 물건만 인식")
    parser.add_argument("--out", default="results", help="결과를 저장할 폴더 (기본 results, 실행할 때마다 안에 새 폴더)")
    args = parser.parse_args()

    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        sys.exit(f"오류: 사진이나 폴더가 없습니다: {', '.join(missing)}")
    pairs = _pairs(args.paths)
    if not pairs:
        sys.exit("오류: 인식할 사진이 없습니다. 폴더에는 사진이 두 장 이상 있어야 합니다. (첫 장은 기준 사진)")

    if not os.environ.get("GEMINI_API_KEY"):  # 결과 폴더를 만들기 전에 확인한다
        sys.exit("오류: GEMINI_API_KEY 환경변수가 없습니다. API 키를 먼저 설정하세요.")

    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    times = []
    with open(out / "인식결과.csv", "w", encoding="utf-8-sig", newline="") as f:  # 엑셀에서 한글이 안 깨지게
        writer = csv.writer(f)
        writer.writerow(["사진", "비교한 사진", "걸린 시간(초)", "번호", "종류", "이름", "색상", "특징", "비공개"])
        for n, (photo, before) in enumerate(pairs, 1):
            print(f"[{n}/{len(pairs)}] {photo.name} " + (f"(앞 사진: {before.name})" if before else "(보이는 물건 전부)"))
            try:
                after_image, before_image = load_image(photo), load_image(before) if before else None
            except ValueError as e:
                print(f"    {e}")
                continue
            items, elapsed = _recognize(after_image, before_image)
            row = [photo.name, before.name if before else ""]
            if items is None:
                writer.writerow(row + ["", "실패"])
                continue
            times.append(elapsed)
            if not items:
                print(f"    {'새 물건 없음' if before else '물건 없음'} ({elapsed:.1f}초)")
                writer.writerow(row + [f"{elapsed:.1f}"])
            else:
                print(f"    물건 {len(items)}개 인식 ({elapsed:.1f}초)")
            for i, item in enumerate(items, 1):
                (out / f"{n:02d}_물건{i}.jpg").write_bytes(item.photo)  # 비공개 물건은 이미 모자이크된 사진이다
                print(f"    [{i}] {item.category} · {item.color} · {item.name}" + (" [비공개]" if item.is_private else ""))
                print(f"        특징: {item.features}")
                writer.writerow(row + [f"{elapsed:.1f}", i, item.category, item.name, item.color, item.features,
                                       "예" if item.is_private else "아니오"])
            f.flush()  # 중간에 꺼도 그때까지의 결과가 남게
    if times:
        print(f"\n사진 {len(times)}장 인식, 한 장에 평균 {sum(times) / len(times):.1f}초 (결과: {out})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C로 꺼도 오류 메시지 없이 끝낸다
        pass
