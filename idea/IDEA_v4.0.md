## 📥 데이터 수집 & 인덱싱 전략

- **일괄 수동 수집**: 공식 PDF / 공식 페이지만 다운로드 → 인덱싱  
- 정부·공공·통신 공식 문서만 사용 → 신뢰성 / 저작권 / 유지보수 모두 안전  
  (예: HiKorea, 1345, Visa Portal, 서울시 영문 가이드, Invest Korea, T-money, KT, LG U+)  

### 주요 공식 소스
- [Hi Korea](https://www.hikorea.go.kr)  
- [Ministry of Justice](https://www.moj.go.kr)  
- [Korea Visa Portal](https://www.visa.go.kr)  
- [Seoul Official](https://english.seoul.go.kr)  
- [InvestKOREA](https://www.investkorea.org)  
- [T-money](https://www.t-money.co.kr)  
- [KT Roaming](https://roaming.kt.com)  
- [LG U+ Global](https://mglobal.lguplus.com)  

---

## 🖥 실행 방식

- 프론트 (Streamlit) · FastAPI: **“있는 걸로 가정”**만 하고 지금은 미구현  
- 대신 CLI / 노트북 실행 → 결과를 **가이드 포스트 JSON/Markdown**으로 바로 출력  

---

## 🚀 킬러 기능 (MVP)

- 질문 → 단계별 **검증된 절차 카드** 생성 (EN ↔ KO)  
- 문서/페이지 출처 자동 첨부  

---

## 📦 포함할 공식 소스 (초기 6~8개)

- **출입국/체류**: HiKorea 포털(공식 안내·예약/전자민원 허브), 1345 다국어 상담  
- **비자 포털**: Visa Navigator & e-Visa 안내  
- **서울 생활**: 2024 Living in Seoul (영문 PDF)  
- **은행 계좌**: Invest Korea, *How Foreigners Can Open a Bank Account in Korea* (2024-08 PDF)  
- **교통카드**: T-money 외국인 안내 (충전 단위 · 잔액 상한 공식)  
- **통신**: KT SIM 유저 가이드(영문), LG U+ 글로벌(외국인용 상품/FAQ)  

> 모든 답변 하단에 **1345 (법무부 출입국) 다국어 상담 안내** 고정 노출  


## 🧠 모델 / 검색 구성 (2일 내 현실 플랜)

- **임베딩**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (다국어 / 경량 / 성능 밸런스)  
- **하이브리드 검색**: BM25 + 벡터(top_k=25) → `bge-reranker-v2-m3`로 top-8 압축  
- **청킹**: 섹션/소제목 기반 800~1,200자, 15% overlap, 페이지 번호 메타 포함  
- **생성 프롬프트**: 근거 문장 내에서만 재진술, 단계별 카드 + `[문서명, p.xx | 연도]` 인용  

### 가드레일

- **출입국·법령 키워드 감지** (visa, sojourn, D-2, work hours 등) 시 자동 첨부:  
  - [HiKorea](https://www.hikorea.go.kr)  
  - [Visa Portal](https://www.visa.go.kr)  
  - [1345 안내 (Ministry of Justice)](https://www.moj.go.kr)  

- **Low confidence** (리랭크 점수↓) → **“모름 + 공식 링크만” 반환** 

## 설치 & 실행 스케치

```
# 1) 필수 패키지
pip install qdrant-client sentence-transformers flagembedding rank-bm25 pydantic pypdf pymupdf langchain-core langchain-community

# 2) Qdrant 로컬 실행
docker run -p 6333:6333 -v $(pwd)/index:/qdrant/storage qdrant/qdrant

# 3) 데이터 수집 (공식 URL만)
python scripts/00_fetch.py

# 4) PDF 파싱 → JSON
python scripts/01_parse_pdf.py

# 5) 인덱싱
python scripts/02_index_qdrant.py

# 6) 질의 → 가이드 생성 (EN/KO 동시)
python scripts/03_answer.py --q "How can I extend my student (D-2) visa?"
```

## 🔑 핵심 코드 포인트 (요약)

- **파싱**: PyMuPDF → `{ "text", "page", "source", "year", "lang" }`  
- **인덱스**: `bge-m3.encode()` → Qdrant payload에 페이지/연도 저장  
- **검색**: BM25 + 벡터 스코어 정규화 가중합 → top-50 후보  
- **리랭킹**: `bge-reranker`로 top-8 압축  
- **생성**: 단계 카드(제목/본문/근거칩) EN↔KO 동시 출력 + 하단 고정 링크 (HiKorea / 1345 / Visa Portal)  

---

## 🗓️ 2일 집중 플랜 (총 14~16h)

**Day 1**
- (3h) PDF/페이지 파싱 + 비정상 문자 정규화  
- (2h) 임베딩/인덱싱 (bge-m3 + Qdrant)  

**Day 2**
- (3h) 하이브리드 검색 + 리랭커 파이프라인 완성  
- (2h) 절차 카드 생성 프롬프트/포맷터 (EN↔KO)  
- (1h) 가드레일 구현 (1345/HiKorea/Visa Portal 자동 첨부)  
- (1h) 샘플 10문항 스모크 테스트  
---

## 🔎 포함할 고정 링크 (최소 셋)

- [HiKorea](https://www.hikorea.go.kr)  
- [Visa Portal](https://www.visa.go.kr)  
- [Seoul Official – Living in Seoul 2024](https://english.seoul.go.kr/wp-content/uploads/2022/07/2024_Living_in_Seoul.pdf)  
- [InvestKOREA – Bank Account Guide](https://www.investkorea.org)  
- [T-money](https://www.t-money.co.kr)  
- [KT Roaming](https://roaming.kt.com)  
- [LG U+ Global](https://mglobal.lguplus.com)  
