# 🎯 외국인 생활 가이드 에이전트 (절차 카드형 MVP)

## ✅ 핵심 메리트
질문을 하면, 에이전트가 **공식 문서에서만 근거를 뽑아** 하나의 “검증된 가이드 글”을 생성.  

- **① 단계별 절차** (필요서류 · 예약 · 수수료 · 처리시간)  
- **② 페이지 단위 인용** (문서명 · 쪽 · 링크)  
- **③ EN ↔ KO 동시 제공**  

→ 최종적으로 **Steemit-style 포스트**로 발행, 링크 공유·북마크 가능.  
(정보 재사용성↑, 신뢰성↑)

---

## 📖 예시: D-2 비자 연장
- HiKorea 전자신청/예약 경로 + 단계별 절차 카드화  
- 각 단계 옆에 **근거 칩**: `[문서명, p.xx | 연도] + [원문 열기]`  
- 하단 고정: **Immigration 1345 다국어 상담 안내**  

---

## 🧱 프론트엔드 (스팀잇 미니멀 스타일)

- **홈(피드)**: 최신/인기 가이드 카드 리스트 (제목 · 요약 · 최종검증일 · 태그)  
- **포스트 상세**  
  - 상단: 요약(EN/KO 토글), “최종 확인일 YYYY-MM-DD”  
  - 본문: Step 1~6 카드 (각 카드 옆 근거칩: 문서명, 쪽, 링크)  
  - 하단: 1345 안내 + 관련 공식 링크  
- **작성(Ask)**: 질문 입력 → 가이드 글 생성 미리보기 → 발행  
- **UI 레이아웃**: 카드 1열 + 넓은 본문  
- MVP에서 댓글/계정 제외(북마크만 지원)

---

## ⚙️ 백엔드 / RAG 설계

- **임베딩**: multilingual-e5-base 또는 bge-m3 (다국어 지원, 비용/성능 밸런스)  
- **리트리버**: BM25 + 벡터 하이브리드 → `bge-reranker-m3` (top-50 → top-8)  
- **청킹**: 섹션·목차 기준 800~1,200자 (15% overlap)  
- **응답 생성**: 근거 청크 내 문장만 사용 (추측 금지)  
- **신선도 전략**: 최신 문서 가중치 ↑, 오래된 문서는 페널티 ↓  
- **가드레일**: 출입국/법·제도 관련 → 항상 HiKorea, Visa Portal, 1345 링크 자동 추가  

---

## 📚 초기 공식 소스 (4종)

1. **출입국**  
   - HiKorea 포털 (안내/예약/전자신청) → [Hi Korea](https://www.hikorea.go.kr)  
   - Visa Portal (전자비자) → [Korea Visa Portal](https://www.visa.go.kr)  

2. **생활**  
   - 서울시 Living in Seoul 2024 (PDF) → [Seoul Official](https://english.seoul.go.kr)  

3. **은행**  
   - Invest Korea – How Foreigners Can Open a Bank Account (2024-08 PDF)  
     → [InvestKOREA](https://www.investkorea.org)  

4. **상담**  
   - 법무부 출입국 1345 (다국어 상담, 운영시간 안내)  
     → [Ministry of Justice](https://www.moj.go.kr)  

---

## 🗂 데이터 모델 (예시)

```
{
  "id": "post_20250822_visa_extend_d2",
  "title": "D-2 Visa Extension (유학생 체류기간 연장)",
  "lang": ["en","ko"],
  "updated_at": "2025-08-22",
  "tags": ["immigration","visa","student"],
  "summary": {
    "en": "...",
    "ko": "..."
  },
  "steps": [
    {
      "title": "Reserve or e-Apply on HiKorea",
      "body": {"en": "...", "ko": "..."},
      "citations": [
        {"source": "HiKorea", "page": "Guide", "url": "..."}
      ]
    }
  ],
  "always_show": [
    {
      "label": "Immigration Contact Center 1345",
      "url": "...",
      "note": "multilingual"
    }
  ]
}
```

## 🗓 MVP 개발 범위 (2~4주)

- 가이드 글 생성기 (킬러 기능): 질문 → 절차 카드 → 포스트 발행  
- EN/KO 토글 + 페이지 단위 근거 칩  
- 피드/상세/북마크  
- 데이터 소스 4종 고정 (출입국/비자, 서울생활, 계좌, 1345)  
- 추측 금지: 불확실 → “모름 + 공식 링크”  

---

## 📌 왜 이게 먹히나?

- **“완성된 가이드 글”**이 바로 생김 → 카톡/단톡 공유 용이  
- 공식 문서만 인용 → 신뢰성·재현성 확보  
- 프론트/백 단순 → 1인 유지 가능, 확장(언어·도시) 점진적  

---

## 📚 핵심 링크 모음

- [HiKorea](https://www.hikorea.go.kr)  
- [Korea Visa Portal](https://www.visa.go.kr)  
- [Seoul Official](https://english.seoul.go.kr)  
- [InvestKOREA](https://www.investkorea.org)  
- [Ministry of Justice](https://www.moj.go.kr)  