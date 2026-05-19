"""
MIRACLE INTEL v5.4 - 미라클탐정사무소 공식 AI 민간조사 플랫폼
무제한 통합 검색 엔진 - 모든 정보 수집
대표: 최다슬 탐정
"""
import os, json, re, httpx, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

AI_PROVIDER         = os.getenv("AI_PROVIDER", "groq").lower()
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")
KAKAO_API_KEY       = os.getenv("KAKAO_API_KEY", "")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID       = os.getenv("GOOGLE_CSE_ID", "")
DART_API_KEY        = os.getenv("DART_API_KEY", "")
DATA_API_KEY        = os.getenv("DATA_API_KEY", "")

GROQ_MODEL      = "llama-3.3-70b-versatile"
GEMINI_MODEL    = "gemini-2.5-flash-preview-05-20"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

app = FastAPI(title="MIRACLE INTEL API v5.4", version="5.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── 모델 ──────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system: str = ""
    max_tokens: int = 2000

class SearchRequest(BaseModel):
    query: str
    search_type: str = "all"
    display: int = 20

class OsintRequest(BaseModel):
    query: str
    report_type: str = "full"

class SimpleRequest(BaseModel):
    query: str

class DartRequest(BaseModel):
    corp_name: str = ""
    corp_code: str = ""
    report_type: str = "all"

class BizRequest(BaseModel):
    b_no: str  # 사업자등록번호 10자리 (하이픈 없이)

class LawRequest(BaseModel):
    query: str
    law_type: str = "all"

class MethodRequest(BaseModel):
    query: str
    method_type: str = "all"

class MediaRequest(BaseModel):
    query: str
    media_type: str = "all"
    accuracy: str = "high"

class ReportRequest(BaseModel):
    client_name: str
    client_tel: str = ""
    client_addr: str = ""
    case_type: str
    case_id: str
    period: str = ""
    target: str = ""
    purpose: str = ""
    detail: str
    result: str = ""

# ─── AI 호출 ───────────────────────────────────────────────────
async def call_ai(messages: list, system: str = "", max_tokens: int = 2000) -> str:
    p = AI_PROVIDER
    if p == "auto":
        if GROQ_API_KEY:        p = "groq"
        elif GEMINI_API_KEY:    p = "gemini"
        elif ANTHROPIC_API_KEY: p = "anthropic"
        else: raise HTTPException(500, "API 키 없음")
    if p == "groq":      return await call_groq(messages, system, max_tokens)
    elif p == "gemini":  return await call_gemini(messages, system, max_tokens)
    elif p == "anthropic": return await call_anthropic(messages, system, max_tokens)
    raise HTTPException(500, f"알 수 없는 프로바이더: {p}")

async def call_groq(messages, system, max_tokens):
    if not GROQ_API_KEY: raise HTTPException(500, "GROQ_API_KEY 미설정")
    msgs = []
    if system: msgs.append({"role":"system","content":system})
    msgs.extend(messages)
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
            json={"model":GROQ_MODEL,"messages":msgs,"max_tokens":max_tokens,"temperature":0.3})
        if r.status_code != 200: raise HTTPException(r.status_code, f"Groq 오류: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]

async def call_gemini(messages, system, max_tokens):
    if not GEMINI_API_KEY: raise HTTPException(500, "GEMINI_API_KEY 미설정")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    contents = [{"role":"model" if m["role"]=="assistant" else "user","parts":[{"text":m["content"]}]} for m in messages]
    payload = {"contents":contents,"generationConfig":{"maxOutputTokens":max_tokens,"temperature":0.3}}
    if system: payload["systemInstruction"] = {"parts":[{"text":system}]}
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post(url, json=payload)
        if r.status_code != 200: raise HTTPException(r.status_code, f"Gemini 오류: {r.text[:300]}")
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

async def call_anthropic(messages, system, max_tokens):
    if not ANTHROPIC_API_KEY: raise HTTPException(500, "ANTHROPIC_API_KEY 미설정")
    payload = {"model":ANTHROPIC_MODEL,"max_tokens":max_tokens,"messages":messages}
    if system: payload["system"] = system
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json","x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01"},
            json=payload)
        if r.status_code != 200: raise HTTPException(r.status_code, f"Anthropic 오류: {r.text[:300]}")
        return "\n".join(b.get("text","") for b in r.json().get("content",[]) if b.get("type")=="text")

def parse_json(text: str) -> dict:
    clean = re.sub(r'```(?:json)?','',text).strip().rstrip('`').strip()
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    try: return json.loads(clean)
    except: return {}

def clean_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text or '')

# ─── 네이버 검색 (무제한) ────────────────────────────────────────
async def search_naver(query: str, stype: str = "news", display: int = 20) -> list:
    if not NAVER_CLIENT_ID: return []
    type_map = {
        "news": "news", "web": "webkr", "blog": "blog",
        "cafe": "cafearticle", "image": "image", "doc": "doc",
        "book": "book", "encyc": "encyc", "movie": "movie",
        "local": "local", "kin": "kin"
    }
    st = type_map.get(stype, stype)
    url = f"https://openapi.naver.com/v1/search/{st}.json"
    headers = {"X-Naver-Client-Id":NAVER_CLIENT_ID,"X-Naver-Client-Secret":NAVER_CLIENT_SECRET}
    params = {"query":query,"display":min(display,100),"sort":"date"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=headers, params=params)
            if r.status_code != 200: return []
            items = r.json().get("items", [])
            results = []
            for item in items:
                title = clean_html_tags(item.get("title",""))
                desc  = clean_html_tags(item.get("description","") or item.get("comment",""))
                results.append({
                    "source": "네이버", "type": stype,
                    "title": title, "description": desc,
                    "url": item.get("link","") or item.get("originallink",""),
                    "date": item.get("pubDate","") or item.get("postdate",""),
                    "publisher": item.get("bloggername","") or item.get("cafename","") or ""
                })
            return results
    except: return []

# ─── 카카오 검색 (무제한) ────────────────────────────────────────
async def search_kakao(query: str, stype: str = "web", page: int = 1, size: int = 10) -> list:
    if not KAKAO_API_KEY: return []
    type_map = {"web":"web","news":"news","blog":"blog","cafe":"cafe","image":"image","vclip":"vclip","book":"book"}
    st = type_map.get(stype, "web")
    url = f"https://dapi.kakao.com/v2/search/{st}"
    headers = {"Authorization":f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query":query,"page":page,"size":min(size,50),"sort":"recency"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers=headers, params=params)
            if r.status_code != 200: return []
            docs = r.json().get("documents", [])
            results = []
            for doc in docs:
                title = clean_html_tags(doc.get("title","") or doc.get("name",""))
                desc  = clean_html_tags(doc.get("contents","") or doc.get("preview","") or "")
                results.append({
                    "source": "카카오", "type": stype,
                    "title": title, "description": desc,
                    "url": doc.get("url","") or doc.get("docurl",""),
                    "date": doc.get("datetime","") or doc.get("postdate",""),
                    "publisher": doc.get("blogname","") or doc.get("cafename","") or doc.get("publisher","") or ""
                })
            return results
    except: return []

# ─── 구글 검색 (무제한) ──────────────────────────────────────────
async def search_google(query: str, num: int = 10, start: int = 1) -> list:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID: return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key":GOOGLE_API_KEY,"cx":GOOGLE_CSE_ID,"q":query,"num":min(num,10),"start":start,"lr":"lang_ko","gl":"kr","safe":"off"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200: return []
            items = r.json().get("items", [])
            results = []
            for item in items:
                results.append({
                    "source": "구글", "type": "web",
                    "title": item.get("title",""),
                    "description": item.get("snippet",""),
                    "url": item.get("link",""),
                    "date": "",
                    "publisher": item.get("displayLink","")
                })
            return results
    except: return []

# ─── DART 금감원 기업공시 ─────────────────────────────────────────
async def search_dart_corp(corp_name: str) -> dict:
    """DART 기업코드 조회"""
    if not DART_API_KEY: return {}
    url = "https://opendart.fss.or.kr/api/company.json"
    params = {"crtfc_key": DART_API_KEY, "corp_name": corp_name, "page_no": 1, "page_count": 10}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200: return {}
            data = r.json()
            if data.get("status") != "000": return {}
            return data
    except: return {}

async def search_dart_disclosure(corp_code: str, bgn_de: str = "", end_de: str = "") -> dict:
    """DART 공시 목록 조회"""
    if not DART_API_KEY: return {}
    url = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "page_no": 1,
        "page_count": 20,
        "sort": "date",
        "sort_mth": "desc"
    }
    if bgn_de: params["bgn_de"] = bgn_de
    if end_de: params["end_de"] = end_de
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200: return {}
            data = r.json()
            if data.get("status") != "000": return {}
            return data
    except: return {}

async def search_dart_company_info(corp_code: str) -> dict:
    """DART 기업 기본정보 조회"""
    if not DART_API_KEY: return {}
    url = "https://opendart.fss.or.kr/api/company.json"
    params = {"crtfc_key": DART_API_KEY, "corp_code": corp_code}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200: return {}
            data = r.json()
            if data.get("status") != "000": return {}
            return data
    except: return {}

# ─── 국세청 사업자등록정보 진위확인 ──────────────────────────────────
async def check_business(b_no: str) -> dict:
    """국세청 사업자등록 진위확인 및 상태조회"""
    if not DATA_API_KEY: return {"error": "DATA_API_KEY 미설정"}
    b_no_clean = re.sub(r'[^0-9]', '', b_no)
    if len(b_no_clean) != 10:
        return {"error": "사업자등록번호는 10자리 숫자여야 합니다"}
    url = "https://api.odcloud.kr/api/nts-businessman/v1/status"
    params = {"serviceKey": DATA_API_KEY}
    payload = {"b_no": [b_no_clean]}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, params=params, json=payload,
                             headers={"Content-Type": "application/json"})
            if r.status_code != 200:
                return {"error": f"API 오류: {r.status_code}", "detail": r.text[:200]}
            data = r.json()
            items = data.get("data", [])
            if not items:
                return {"error": "조회 결과 없음", "raw": data}
            item = items[0]
            tax_type_map = {
                "01": "일반과세자", "02": "간이과세자", "03": "면세사업자",
                "04": "비영리법인", "05": "국가지방자치단체", "06": "면세사업자(공동사업자)",
                "07": "영세율 등 조기환급", "08": "기타"
            }
            b_stt_map = {
                "01": "계속사업자", "02": "휴업자", "03": "폐업자"
            }
            return {
                "b_no": b_no_clean,
                "b_no_formatted": f"{b_no_clean[:3]}-{b_no_clean[3:5]}-{b_no_clean[5:]}",
                "b_stt": b_stt_map.get(item.get("b_stt_cd", ""), item.get("b_stt", "알 수 없음")),
                "b_stt_cd": item.get("b_stt_cd", ""),
                "tax_type": tax_type_map.get(item.get("tax_type_cd", ""), item.get("tax_type", "알 수 없음")),
                "tax_type_cd": item.get("tax_type_cd", ""),
                "end_dt": item.get("end_dt", ""),
                "utcc_yn": item.get("utcc_yn", ""),
                "tax_type_change_dt": item.get("tax_type_change_dt", ""),
                "invoice_apply_dt": item.get("invoice_apply_dt", ""),
                "rbf_tax_type": item.get("rbf_tax_type", ""),
                "rbf_tax_type_cd": item.get("rbf_tax_type_cd", ""),
                "status": "정상조회",
                "source": "국세청 사업자등록정보 진위확인 API"
            }
    except Exception as e:
        return {"error": f"요청 실패: {str(e)}"}

async def validate_business(b_no: str, p_nm: str = "", p_nm2: str = "",
                            b_nm: str = "", start_dt: str = "") -> dict:
    """국세청 사업자등록 진위확인 (상세)"""
    if not DATA_API_KEY: return {"error": "DATA_API_KEY 미설정"}
    b_no_clean = re.sub(r'[^0-9]', '', b_no)
    url = "https://api.odcloud.kr/api/nts-businessman/v1/validate"
    params = {"serviceKey": DATA_API_KEY}
    payload = {"businesses": [{
        "b_no": b_no_clean,
        "start_dt": start_dt,
        "p_nm": p_nm,
        "p_nm2": p_nm2,
        "b_nm": b_nm,
        "corp_no": "",
        "b_sector": "",
        "b_type": ""
    }]}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(url, params=params, json=payload,
                             headers={"Content-Type": "application/json"})
            if r.status_code != 200:
                return {"error": f"API 오류: {r.status_code}"}
            return r.json()
    except Exception as e:
        return {"error": f"요청 실패: {str(e)}"}

# ─── 통합 검색 (모든 소스 병렬) ─────────────────────────────────
async def search_all(query: str, display: int = 20) -> dict:
    tasks = [
        search_naver(query, "news",    display),
        search_naver(query, "web",     display),
        search_naver(query, "blog",    display),
        search_naver(query, "cafe",    display),
        search_naver(query, "doc",     10),
        search_naver(query, "kin",     10),
        search_kakao(query, "news",    1, 10),
        search_kakao(query, "web",     1, 10),
        search_kakao(query, "blog",    1, 10),
        search_kakao(query, "cafe",    1, 10),
        search_google(query, 10, 1),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    def safe(r): return r if isinstance(r, list) else []
    return {
        "naver_news":  safe(results[0]),
        "naver_web":   safe(results[1]),
        "naver_blog":  safe(results[2]),
        "naver_cafe":  safe(results[3]),
        "naver_doc":   safe(results[4]),
        "naver_kin":   safe(results[5]),
        "kakao_news":  safe(results[6]),
        "kakao_web":   safe(results[7]),
        "kakao_blog":  safe(results[8]),
        "kakao_cafe":  safe(results[9]),
        "google":      safe(results[10]),
        "total": sum(len(safe(r)) for r in results)
    }

# ─── 엔드포인트 ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(p.read_text("utf-8") if p.exists() else "<h1>index.html 없음</h1>")

@app.get("/health")
async def health():
    return {
        "status":"정상","version":"5.4.0","provider":AI_PROVIDER,
        "apis":{"naver":bool(NAVER_CLIENT_ID),"kakao":bool(KAKAO_API_KEY),
                "google":bool(GOOGLE_API_KEY),"groq":bool(GROQ_API_KEY),"gemini":bool(GEMINI_API_KEY)}
    }

@app.get("/api/status")
async def api_status():
    return {
        "current": AI_PROVIDER,
        "search_apis": {
            "naver": {"configured":bool(NAVER_CLIENT_ID),"daily_limit":25000,"types":["news","web","blog","cafe","doc","kin"]},
            "kakao": {"configured":bool(KAKAO_API_KEY),"daily_limit":300000,"types":["news","web","blog","cafe"]},
            "google": {"configured":bool(GOOGLE_API_KEY),"daily_limit":100,"types":["web"]}
        }
    }

@app.post("/api/search")
async def integrated_search(req: SearchRequest):
    """무제한 통합 실시간 검색 - 네이버+카카오+구글 전체"""
    if not req.query: raise HTTPException(400, "검색어를 입력하세요")
    results = await search_all(req.query, req.display)
    return results

@app.post("/api/search/naver")
async def naver_search(req: SearchRequest):
    """네이버 단독 검색"""
    types = ["news","web","blog","cafe","doc","kin"] if req.search_type == "all" else [req.search_type]
    tasks = [search_naver(req.query, t, req.display) for t in types]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_items = []
    for r in results:
        if isinstance(r, list): all_items.extend(r)
    return {"query":req.query,"total":len(all_items),"results":all_items}

@app.post("/api/search/kakao")
async def kakao_search(req: SearchRequest):
    """카카오 단독 검색"""
    types = ["news","web","blog","cafe"] if req.search_type == "all" else [req.search_type]
    tasks = [search_kakao(req.query, t, 1, req.display) for t in types]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_items = []
    for r in results:
        if isinstance(r, list): all_items.extend(r)
    return {"query":req.query,"total":len(all_items),"results":all_items}

@app.post("/api/search/google")
async def google_search_ep(req: SearchRequest):
    """구글 단독 검색"""
    items = await search_google(req.query, req.display)
    return {"query":req.query,"total":len(items),"results":items}

@app.post("/api/osint")
async def osint(req: OsintRequest):
    """OSINT 전체 수집 + AI 구역별 보고서"""
    if not req.query: raise HTTPException(400, "조사 대상을 입력하세요")
    search_results = await search_all(req.query, 20)
    total = search_results.get("total", 0)

    # 수집 데이터 요약 (AI 입력용)
    all_items = []
    for key in ["naver_news","naver_web","naver_blog","naver_cafe","kakao_news","kakao_web","google"]:
        all_items.extend(search_results.get(key, []))

    items_text = "\n".join([
        f"[{i['source']}/{i['type']}] {i['title']} | {i.get('date','')[:10]} | {i['url']}\n  {i.get('description','')[:200]}"
        for i in all_items[:30]
    ])

    system = (
        "미라클탐정사무소(대표:최다슬 탐정) OSINT 분석 AI. "
        "수집된 실제 데이터를 바탕으로 상세하고 정확한 보고서 작성. "
        "모든 내용 한국어. 다음 JSON만 반환:\n"
        '{"target":"","riskLevel":"높음/보통/낮음","totalSources":0,'
        '"sections":{'
        '"언론_뉴스":{"title":"언론·뉴스","findings":[{"title":"","description":"","url":"","date":"","source":"","credibility":"높음"}],"summary":""},'
        '"온라인_커뮤니티":{"title":"온라인·커뮤니티","findings":[{"title":"","description":"","url":"","date":"","source":"","credibility":"보통"}],"summary":""},'
        '"공식_정보":{"title":"공식·공공 정보","findings":[{"title":"","description":"","url":"","date":"","source":"","credibility":"높음"}],"summary":""},'
        '"SNS_블로그":{"title":"SNS·블로그","findings":[{"title":"","description":"","url":"","date":"","source":"","credibility":"보통"}],"summary":""},'
        '"종합_분석":{"title":"종합 분석","findings":[{"title":"","description":"","url":"","date":"","source":"","credibility":"높음"}],"summary":""}'
        '},'
        '"keyFindings":[""],"legalNotes":[""],"nextSteps":[""],'
        '"reportDate":"","analyst":"미라클탐정사무소 AI 분석 시스템"}'
    )

    text = await call_ai(
        [{"role":"user","content":f"조사 대상: {req.query}\n총 수집: {total}건\n\n수집 데이터:\n{items_text}"}],
        system, 2000
    )
    data = parse_json(text)
    if not data:
        data = {"target":req.query,"riskLevel":"보통","totalSources":total,
                "sections":{},"keyFindings":[text[:500]],"legalNotes":[],"nextSteps":[]}
    data["rawResults"] = search_results
    data["totalSources"] = total
    return data

@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role":m.role,"content":m.content} for m in req.messages]
    system = req.system or (
        "미라클탐정사무소(대표:최다슬 탐정) MIRACLE INTEL v5.4 AI 어시스턴트. "
        "취급업무 21종 전문. 탐정 조사 관련 모든 질문에 상세하고 실용적으로 답변. "
        "한국어로만 답변. 법률 준수 범위 내에서 최대한 자세하게 안내.")
    return {"reply": await call_ai(messages, system, req.max_tokens)}

@app.post("/api/law")
async def law(req: LawRequest):
    law_news = await search_naver(f"{req.query} 법원 판결 법률", "news", 10)
    law_web  = await search_naver(f"{req.query} 법", "web", 5)
    system = (
        "대한민국 법률·판례 전문 분석 AI. 탐정·민간조사 분야. 한국어로만. "
        "수집된 실제 자료를 바탕으로 상세하게 분석. 다음 JSON만 반환:\n"
        '{"query":"","lawUpdates":[{"title":"","content":"","date":"","source":"","url":""}],'
        '"verdicts":[{"case":"","court":"","date":"","summary":"","relevance":"","url":""}],'
        '"lawyerPart":{"title":"변호사 파트","items":[{"article":"","interpretation":"","advice":"","risk":""}]},'
        '"lawmanPart":{"title":"법무사 파트","items":[{"article":"","interpretation":"","advice":"","risk":""}]},'
        '"adminPart":{"title":"행정사 파트","items":[{"article":"","interpretation":"","advice":"","risk":""}]},'
        '"keyPoints":[""]}'
    )
    ctx = "\n".join([f"- {n['title']} ({n['date'][:10]}) {n['url']}\n  {n.get('description','')[:150]}" for n in (law_news+law_web)[:10]])
    text = await call_ai([{"role":"user","content":f"법률 조회: {req.query}\n유형: {req.law_type}\n\n관련 자료:\n{ctx}"}], system, 2000)
    data = parse_json(text)
    if not data:
        data = {"query":req.query,"lawUpdates":[],"verdicts":[],
                "lawyerPart":{"title":"변호사 파트","items":[]},"lawmanPart":{"title":"법무사 파트","items":[]},
                "adminPart":{"title":"행정사 파트","items":[]},"keyPoints":[text[:600]]}
    if law_news and not data.get("lawUpdates"):
        data["lawUpdates"] = [{"title":n["title"],"content":n["description"],"date":n["date"],"source":n["source"],"url":n["url"]} for n in law_news[:5]]
    return data

@app.post("/api/method")
async def method(req: MethodRequest):
    news = await search_naver(f"탐정 {req.query} 조사방법 사례", "blog", 10)
    web  = await search_naver(f"탐정 {req.query}", "web", 5)
    system = (
        "한국 탐정 조사방법론 전문 AI. 실제 수집 자료 기반. 한국어로만. "
        "다음 JSON만 반환:\n"
        '{"query":"","methods":[{"category":"","title":"","description":"","sourceType":"논문/뉴스/블로그/탐정협회/유튜브",'
        '"legalBasis":"","reliability":90,"url":"","isYoutube":false,"channelName":"","videoId":""}],'
        '"keyInsights":[""],"legalNotes":[""],"applicableCases":[""]}'
    )
    ctx = "\n".join([f"- {n['title']}: {n.get('description','')[:150]} {n['url']}" for n in (news+web)[:10]])
    text = await call_ai([{"role":"user","content":f"탐정 방법론: {req.query}\n\n참고자료:\n{ctx}"}], system, 2000)
    data = parse_json(text)
    if not data:
        data = {"query":req.query,"methods":[],"keyInsights":[text[:600]],"legalNotes":[],"applicableCases":[]}
    return data

@app.post("/api/media")
async def media(req: MediaRequest):
    tasks = [
        search_naver(req.query, "news", 10),
        search_naver(req.query, "blog", 5),
        search_naver(req.query, "cafe", 5),
        search_kakao(req.query, "news", 1, 10),
        search_kakao(req.query, "web",  1, 5),
        search_google(req.query, 10),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_items = []
    for r in results:
        if isinstance(r, list): all_items.extend(r)
    sources = [{"channel":i["source"],"outlet":i["publisher"],"title":i["title"],
                "summary":i["description"],"date":i["date"],"reliability":90,
                "url":i["url"],"isYoutube":False,"videoId":""} for i in all_items[:20]]
    return {
        "query":req.query,"accuracy":"높음",
        "sources":sources,
        "keyFacts":[f"총 {len(all_items)}건 수집 (네이버/카카오/구글)"],
        "warnings":[]
    }

@app.post("/api/report")
async def report(req: ReportRequest):
    system = (
        "미라클탐정사무소(대표:최다슬 탐정) 공식 민간조사 보고서 작성 AI. "
        "항목: 1.사건개요 2.조사의뢰 내용 3.조사방법 및 절차 4.조사결과 5.결론 및 권고사항. "
        "전문적이고 상세하게 작성. 한국어로만.")
    text = await call_ai([{"role":"user","content":
        f"의뢰인:{req.client_name}\n사건번호:{req.case_id}\n유형:{req.case_type}\n"
        f"대상:{req.target}\n기간:{req.period}\n목적:{req.purpose}\n내용:{req.detail}"}],
        system, 2000)
    return {"report": text}

# ─── DART 기업공시 엔드포인트 ──────────────────────────────────────

@app.post("/api/dart/search")
async def dart_search(req: DartRequest):
    """DART 기업 검색 + 공시 목록 통합 조회"""
    if not DART_API_KEY:
        raise HTTPException(500, "DART_API_KEY 미설정")
    if not req.corp_name and not req.corp_code:
        raise HTTPException(400, "기업명 또는 기업코드를 입력하세요")

    corp_code = req.corp_code
    corp_info = {}
    corp_list = []

    # 기업명으로 검색
    if req.corp_name and not corp_code:
        result = await search_dart_corp(req.corp_name)
        corp_list = result.get("list", [])
        if corp_list:
            corp_code = corp_list[0].get("corp_code", "")
            corp_info = corp_list[0]

    # 기업코드로 상세정보 조회
    if corp_code:
        info_result = await search_dart_company_info(corp_code)
        if info_result:
            corp_info = info_result

    # 공시 목록 조회
    disclosures = []
    if corp_code:
        disc_result = await search_dart_disclosure(corp_code)
        disclosures = disc_result.get("list", [])

    return {
        "corp_name": req.corp_name,
        "corp_code": corp_code,
        "corp_info": corp_info,
        "corp_list": corp_list[:5],
        "disclosures": disclosures[:20],
        "total_disclosures": len(disclosures),
        "source": "금융감독원 DART 전자공시시스템",
        "dart_configured": bool(DART_API_KEY)
    }

@app.get("/api/dart/company/{corp_code}")
async def dart_company(corp_code: str):
    """DART 기업 기본정보 조회"""
    if not DART_API_KEY:
        raise HTTPException(500, "DART_API_KEY 미설정")
    info = await search_dart_company_info(corp_code)
    disclosures_raw = await search_dart_disclosure(corp_code)
    disclosures = disclosures_raw.get("list", [])
    return {
        "corp_code": corp_code,
        "info": info,
        "recent_disclosures": disclosures[:10],
        "source": "금융감독원 DART"
    }

# ─── 국세청 사업자 조회 엔드포인트 ─────────────────────────────────

@app.post("/api/business/check")
async def business_check(req: BizRequest):
    """국세청 사업자등록 상태조회"""
    if not DATA_API_KEY:
        raise HTTPException(500, "DATA_API_KEY 미설정")
    result = await check_business(req.b_no)
    return result

@app.post("/api/business/validate")
async def business_validate(req: dict):
    """국세청 사업자등록 진위확인 (상세)"""
    if not DATA_API_KEY:
        raise HTTPException(500, "DATA_API_KEY 미설정")
    b_no    = req.get("b_no", "")
    p_nm    = req.get("p_nm", "")
    p_nm2   = req.get("p_nm2", "")
    b_nm    = req.get("b_nm", "")
    start_dt = req.get("start_dt", "")
    if not b_no:
        raise HTTPException(400, "사업자등록번호를 입력하세요")
    result = await validate_business(b_no, p_nm, p_nm2, b_nm, start_dt)
    return result

@app.get("/api/business/status")
async def business_api_status():
    """사업자 API 상태 확인"""
    return {
        "dart_configured": bool(DART_API_KEY),
        "data_configured": bool(DATA_API_KEY),
        "dart_key_preview": DART_API_KEY[:8] + "..." if DART_API_KEY else "미설정",
        "data_key_preview": DATA_API_KEY[:8] + "..." if DATA_API_KEY else "미설정",
    }

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("  MIRACLE INTEL v5.4 - 미라클탐정사무소")
    print("  대표: 최다슬 탐정 | 무제한 통합검색")
    print("="*60)
    print(f"  AI:      {AI_PROVIDER.upper()} {'OK' if (GROQ_API_KEY or GEMINI_API_KEY) else '키 없음'}")
    print(f"  네이버:  {'OK' if NAVER_CLIENT_ID else '미설정'}")
    print(f"  카카오:  {'OK' if KAKAO_API_KEY else '미설정'}")
    print(f"  구글:    {'OK' if GOOGLE_API_KEY else '미설정'}")
    print(f"  DART:    {'OK' if DART_API_KEY else '미설정'}")
    print(f"  사업자:  {'OK' if DATA_API_KEY else '미설정'}")
    print(f"  서버:    http://localhost:8000")
    print("="*60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="warning")
