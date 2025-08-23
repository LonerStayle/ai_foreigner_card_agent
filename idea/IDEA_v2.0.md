# 🌍 외국인 생활 가이드 에이전트 (MVP)

## ✅ 한 줄 요약
- **범위**: 출입국/체류, 은행계좌, 통신(SIM/eSIM), 대중교통 카드(T-money)  
- **데이터**: 정부·공공·기관 **공식 문서(PDF/가이드)**만 인덱싱  
- **기능**: 다국어(영/한) RAG 답변 + 문서/섹션 출처 + “원문 바로가기”  
- **가드레일**: 법·제도 관련 질의 → 항상 **공식 링크·전화 1345** 병기  
- **UI**: 질문 → 요약 답변 → 근거(문서/페이지) → 원문 링크 → (선택) 자동 번역 토글.  

---

## 📦 데이터 소스 (초기 6종)

### 1. 출입국·체류
- HiKorea (법무부 출입국 포털) – 비자·체류 안내, 방문예약, 전자민원  
  👉 [Hi Korea](https://www.hikorea.go.kr)  
- 법무부 출입국 영문 사이트 – 공지/비자 내비게이터 가이드  
  👉 [Ministry of Justice](https://www.moj.go.kr)  
- (확장 시) 비자 포털 e-Visa 절차  
  👉 [Korea Visa Portal](https://www.visa.go.kr)  

### 2. 생활가이드
- 서울시 **Living in Seoul 2024 (PDF)** – 행정·생활절차 총괄 안내  
  👉 [Seoul Official](https://english.seoul.go.kr)  

### 3. 은행 계좌 개설
- KOTRA / Invest Korea – **How Foreigners Can Open a Bank Account** (PDF)  
  👉 [InvestKOREA](https://www.investkorea.org)  

### 4. 통신
- KT SIM 사용자 가이드 (영문)  
  👉 [KT Roaming](https://roaming.kt.com)  
- LG U+ Global (다국어 지원/상담)  
  👉 [LG U+ Global](https://mglobal.lguplus.com)  
- (참고) SKT eSIM 안내  
  👉 [SK Telecom Roaming](https://www.skroaming.com)  

### 5. 대중교통
- T-money 공식 외국인 안내 (충전 단위·상한 등)  
  👉 [T-money](https://www.t-money.co.kr)  

### 6. 참고
- MCIA South Korea Country Handbook (PDF)  
  → **법·제도 근거 아님**, 참고자료로만 사용  

---

## 🔎 사용자 시나리오 → 소스 매핑

- **“How can I extend my visa?”**  
  → HiKorea 체류연장 안내 + 전자신청/예약 + 최신 공지  

- **“한국에서 외국인이 은행 계좌 개설하려면?”**  
  → Invest Korea 가이드 요약 + 은행별 링크  

- **“유학생 아르바이트 규정?”**  
  → HiKorea D-2 비자 ‘시간제취업확인서’ + 비허용 직종 안내 + 1345 병기  

- **“T-money 충전 방법”**  
  → 공식 안내: 충전 단위/상한, 편의점·자판기 충전  

- **“통신사 가입/유심 개통”**  
  → KT/LG U+ 외국인 안내 페이지, 상담번호, 영업시간  

---

## 🏗 제품 아키텍처 (MVP)

### 백엔드
- API: **FastAPI (Python)**
- 임베딩: **bge-m3** (또는 multilingual-e5-base)  
- 벡터DB: **Qdrant** (Docker → 클라우드 확장 가능)  
- 검색: **BM25 + Vector Hybrid**
- 리랭커: **bge-reranker-v2-m3** (다국어)  
- 생성: LLM + 출처 인라인 주석 (문서명/페이지)  

### 프런트엔드
- **Next.js + Tailwind**  
- 카드형 UI: 질문 → 요약 → 근거(문서/페이지) → [원문 보기]  
- EN↔KO 언어 토글  

### 가드레일
- 출입국/법 규정 질의 → 항상 공식 링크 + 1345 안내 병기  
- 신뢰도 낮을 시 “모름 + 원문 링크”  

---

## 🗓 개발 플랜 (4주, 1인)

- **W1** 데이터 & 인덱싱  
  HiKorea / 법무부 / 서울시 / InvestKorea / KT / LG U+ / T-money  

- **W2** 검색·리랭크·생성  
  하이브리드 검색 + 근거 주석 삽입  

- **W3** 프런트 & 번역  
  Next.js UI, EN↔KO 토글, 카드형 근거 UI  

- **W4** 평가·배포  
  100문항 테스트 → 배포(Vercel, Railway, Docker Qdrant)  

---

## 💰 비용 추정 (월)

- 프런트 (Vercel): **$0~20**  
- 백엔드 인스턴스: **$5~10**  
- Qdrant (Docker): **$0~15**  
- LLM 추론: **$10~50**  

---

## 📌 확장 로드맵

1. 자주 묻는 절차 카드 (체류연장/주소변경/재입국)  
2. 대학별 D-2 파트타임 규정 카드  
3. 도시별 생활가이드 추가(부산/인천/대구)  
4. 다국어 확대(중국어/베트남어/러시아어)  

---

## ⚖️ 품질 & 안전

- 응답 내 임의 추정 금지 → “모름” + 공식 링크  
- 정부/공공기관 문서 우선, 블로그 금지  
- 모든 답변 하단에 고지:  
  > “법률 자문 아님, 최신성 변동 가능, 공식 창구 확인 권장 (HiKorea·1345)”  

---

## 📚 핵심 링크 모음
- [HiKorea](https://www.hikorea.go.kr)  
- [Ministry of Justice](https://www.moj.go.kr)  
- [Korea Visa Portal](https://www.visa.go.kr)  
- [Seoul Official](https://english.seoul.go.kr)  
- [InvestKOREA](https://www.investkorea.org)  
- [KT Roaming](https://roaming.kt.com)  
- [LG U+ Global](https://mglobal.lguplus.com)  
- [T-money](https://www.t-money.co.kr)  

