import argparse

# 1. 파서 생성
parser = argparse.ArgumentParser(description="옵션 플래그 예제")

# 2. 옵션 추가 (-- 로 시작하는 게 긴 옵션)
parser.add_argument("--mode", type=str, default="default", help="실행 모드 선택")
parser.add_argument("--q", type=str, required=True, help="쿼리 문장")

# 3. 파싱 실행
args = parser.parse_args()

# 4. 값 사용
print(f"Mode: {args.mode}")
print(f"Query: {args.q}")