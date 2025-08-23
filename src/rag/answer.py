from langchain_openai import ChatOpenAI
from .retriever import hybrid_search
from .reranker import rerank

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature=0)

def generate_answer(query:str):
    candidates = hybrid_search(query, top_k = 50)
    top_docs = rerank(query,candidates)
    context = '\n\n'.join(d['text'] for d in top_docs)
    
    prompt = f"""
    You are a Korean immigration & living guide assistant.
    
    Q: {query}
    
    Context:
    {context}
    
    Output the answer strictly in JSON format with the following schema:
    
    {{
      "steps": [
        {{
          "step": number,
          "title": "short English title",
          "en": "explanation in English",
          "ko": "translation in Korean"
        }}
      ],
      "references": [
        {{ "name": "HiKorea", "url": "https://www.hikorea.go.kr" }},
        {{ "name": "1345 Call Center", "url": null }},
        {{ "name": "Korea Visa Portal", "url": "https://www.visa.go.kr" }}
      ]
    }}
    """
    
    answer = llm.invoke(prompt)
    return answer, top_docs
    
    