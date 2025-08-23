import os
import fitz  # PyMuPDF
import yaml
import json
from Common import Common

SOURCE_FILE = os.path.join(Common.DATA_DIR, "sources.yaml")
OUTPUT_FILE = os.path.join(Common.DATA_DIR, "parsed.jsonl")


def parse_pdf(filepath: str, source_name: str, year: int = None, lang: str = "en"):
    """PDF 파일을 페이지 단위로 파싱해서 JSON 객체 리스트 반환"""
    docs = []
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        print(f"❌ Failed to open {filepath}: {e}")
        return docs

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue

        docs.append({
            "text": text,
            "page": page_num,
            "source": os.path.basename(filepath),
            "name": source_name,
            "year": year,
            "lang": lang
        })
    return docs


def run_parsing():
    """sources.yaml에 정의된 모든 PDF를 파싱 후 JSONL 저장"""
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_docs = []
    for src in config["sources"]:
        filepath = os.path.join(Common.DATA_DIR, src["file"])

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue

        if src["type"] == "pdf":
            docs = parse_pdf(filepath, src["name"], src.get("year"), src.get("lang", "en"))
            all_docs.extend(docs)
            print(f"✅ Parsed {src['name']} → {len(docs)} pages")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n📦 Done. Total {len(all_docs)} chunks saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    run_parsing()