import os
import json
import pathlib
import re
import time
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse, quote_plus

from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import pystache
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import openai
from pytrends.request import TrendReq

# Note: Procfile should use gunicorn with --timeout 90 to allow for API calls with retries

app = Flask(__name__)
CORS(app)  # Enable CORS for n8n

TEMPLATE_PATH = pathlib.Path("templates/nykra_report.html").resolve()

# ---------- HTTP Sessions with Retries ----------
def create_session(prefix: str, connect_timeout: int, read_timeout: int) -> requests.Session:
    """Create a requests session with retries and timeouts"""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Log session creation for debugging
    print(f"[{prefix}] Created session with ({connect_timeout}, {read_timeout}) timeouts")
    return session

# Create service-specific sessions
PSI_SESSION = create_session("PSI", 5, 12)
SERPSTACK_SESSION = create_session("SERPSTACK", 5, 10)

# ---------- Handlebars → Mustache normalizer ----------
def normalize_handlebars_each_to_mustache(html: str) -> str:
    """
    Convert {{#each foo.bar}} ... {{/each}} blocks into Mustache sections:
    -> {{#foo.bar}} ... {{/foo.bar}}
    Works with nested each blocks via a small stack.
    """
    out, i, stack = [], 0, []
    token_re = re.compile(r"(\{\{\#each\s+[^}]+\}\}|\{\{\/each\}\})")
    for m in token_re.finditer(html):
        out.append(html[i:m.start()])
        tok = m.group(0)
        if tok.startswith("{{#each"):
            name = tok[len("{{#each"): -2].strip()
            stack.append(name)
            out.append("{{#" + name + "}}")
        else:  # {{/each}}
            name = stack.pop() if stack else ""
            out.append("{{/" + name + "}}")
        i = m.end()
    out.append(html[i:])
    return "".join(out)

# ---------------------------
# API Integrations
# ---------------------------
def psi_fetch(url: str, strategy: str, api_key: str) -> dict:
    """Fetch Core Web Vitals from Google PageSpeed Insights (robust INP extraction)."""
    default_response = {
        "lcp_ms": 0, "lcp_status": "fail",
        "inp_ms": 0, "inp_status": "fail",
        "cls": 0.0, "cls_status": "fail"
    }
    
    try:
        resp = PSI_SESSION.get(
            "https://www.googleapis.com/pagespeedonline/v5/runPagespeed",
            params={"url": url, "strategy": strategy, "category": "PERFORMANCE", "key": api_key},
            timeout=(5, 12)  # (connect_timeout, read_timeout)
        )
        resp.raise_for_status()
        j = resp.json()
        audits = j.get("lighthouseResult", {}).get("audits", {})

        def num(key):
            v = audits.get(key, {}).get("numericValue")
            return float(v) if isinstance(v, (int, float)) else None

        lcp = num("largest-contentful-paint")
        # Try real INP first, then fall back gracefully
        inp = (num("experimental-interaction-to-next-paint")
               or num("interaction-to-next-paint")
               or num("total-blocking-time")
               or num("interactive"))
        cls = audits.get("cumulative-layout-shift", {}).get("numericValue", None)

        def status_lcp(v):
            if v is None: return "fail"
            return "pass" if v <= 2500 else "needs improvement" if v <= 4000 else "fail"

        def status_inp(v):
            if v is None: return "fail"
            return "pass" if v <= 200 else "needs improvement" if v <= 500 else "fail"

        def status_cls(v):
            if v is None: return "fail"
            return "pass" if v <= 0.1 else "needs improvement" if v <= 0.25 else "fail"

        return {
            "lcp_ms": int(round(lcp)) if isinstance(lcp, (int, float)) else 0,
            "lcp_status": status_lcp(lcp),
            "inp_ms": int(round(inp)) if isinstance(inp, (int, float)) else 0,
            "inp_status": status_inp(inp),
            "cls": round(cls, 3) if isinstance(cls, (int, float)) else 0.0,
            "cls_status": status_cls(cls),
        }
    except Exception as e:
        print(f"[PSI] API error ({strategy}): {e}")
        return default_response

def psi_fetch_parallel(url: str, api_key: str) -> Tuple[dict, dict]:
    """Fetch PSI data for mobile and desktop in parallel using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        mobile_future = executor.submit(psi_fetch, url, "mobile", api_key)
        desktop_future = executor.submit(psi_fetch, url, "desktop", api_key)
        
        # Get results, with timeout protection
        try:
            mobile_result = mobile_future.result(timeout=15)
        except (concurrent.futures.TimeoutError, Exception) as e:
            print(f"[PSI] Mobile fetch timed out or failed: {e}")
            mobile_result = {
                "lcp_ms": 0, "lcp_status": "fail",
                "inp_ms": 0, "inp_status": "fail",
                "cls": 0.0, "cls_status": "fail"
            }
            
        try:
            desktop_result = desktop_future.result(timeout=15)
        except (concurrent.futures.TimeoutError, Exception) as e:
            print(f"[PSI] Desktop fetch timed out or failed: {e}")
            desktop_result = {
                "lcp_ms": 0, "lcp_status": "fail",
                "inp_ms": 0, "inp_status": "fail",
                "cls": 0.0, "cls_status": "fail"
            }
    
    return mobile_result, desktop_result

def serpstack_fetch_keywords(domain: str, api_key: str, limit: int = 10) -> List[Dict]:
    """Fetch top ranking keywords for a domain using Serpstack API."""
    try:
        # Extract domain without protocol
        parsed_domain = urlparse(domain)
        clean_domain = parsed_domain.netloc or parsed_domain.path.split('/')[0]
        
        # Fetch organic results for the domain
        resp = SERPSTACK_SESSION.get(
            "http://api.serpstack.com/search",
            params={
                "access_key": api_key,
                "query": f"site:{clean_domain}",
                "num": limit,
                "output": "json"
            },
            timeout=(5, 10)  # (connect_timeout, read_timeout)
        )
        resp.raise_for_status()
        
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            print(f"[SERPSTACK] JSON decode error: {e}")
            return []
        
        # Extract keywords from organic results
        keywords = []
        for result in data.get("organic_results", [])[:limit]:
            # For each result, get the main keyword
            title = result.get("title", "")
            url = result.get("url", "")
            position = result.get("position", 0)
            
            # Extract path from URL for internal reference
            path = urlparse(url).path or "/"
            
            # Estimate volume based on position (this is a rough approximation)
            # In a real implementation, you'd use a keyword volume API
            volume = max(3000 - (position * 200), 100)
            
            # Extract main keyword from title
            keyword = title.split(" - ")[0].strip()
            if len(keyword) > 5:  # Only include meaningful keywords
                keywords.append({
                    "kw": keyword[:50],  # Limit length
                    "pos": position,
                    "vol": volume,
                    "url": path
                })
        
        return keywords[:limit]
    except Exception as e:
        print(f"[SERPSTACK] API error for domain {domain}: {e}")
        return []

def fetch_keywords_parallel(client_url: str, comp1_url: Optional[str], comp2_url: Optional[str], 
                           api_key: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Fetch keywords for client and competitors in parallel."""
    client_keywords = []
    comp1_keywords = []
    comp2_keywords = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Always fetch client keywords
        client_future = executor.submit(serpstack_fetch_keywords, client_url, api_key)
        
        # Conditionally fetch competitor keywords
        comp1_future = None
        comp2_future = None
        
        if comp1_url:
            comp1_future = executor.submit(serpstack_fetch_keywords, comp1_url, api_key)
        
        if comp2_url:
            comp2_future = executor.submit(serpstack_fetch_keywords, comp2_url, api_key)
        
        # Get results with timeout protection
        try:
            client_keywords = client_future.result(timeout=12)
        except (concurrent.futures.TimeoutError, Exception) as e:
            print(f"[SERPSTACK] Client keywords fetch timed out or failed: {e}")
            client_keywords = []
        
        if comp1_future:
            try:
                comp1_keywords = comp1_future.result(timeout=12)
            except (concurrent.futures.TimeoutError, Exception) as e:
                print(f"[SERPSTACK] Competitor 1 keywords fetch timed out or failed: {e}")
                comp1_keywords = []
        
        if comp2_future:
            try:
                comp2_keywords = comp2_future.result(timeout=12)
            except (concurrent.futures.TimeoutError, Exception) as e:
                print(f"[SERPSTACK] Competitor 2 keywords fetch timed out or failed: {e}")
                comp2_keywords = []
    
    return client_keywords, comp1_keywords, comp2_keywords

def get_keyword_gaps(client_keywords: List[Dict], comp1_keywords: List[Dict], 
                    comp2_keywords: List[Dict]) -> List[Dict]:
    """Find keyword gaps where competitors rank but client doesn't."""
    client_kws = {item["kw"].lower() for item in client_keywords}
    
    gaps = []
    # Check competitor 1 keywords
    for kw_item in comp1_keywords:
        kw = kw_item["kw"].lower()
        if kw not in client_kws:
            gap = {
                "kw": kw_item["kw"],
                "client_rank": "—",
                "comp1_rank": kw_item["pos"],
                "comp2_rank": "—"
            }
            
            # Check if comp2 also ranks for this keyword
            for c2_item in comp2_keywords:
                if c2_item["kw"].lower() == kw:
                    gap["comp2_rank"] = c2_item["pos"]
                    break
                    
            gaps.append(gap)
    
    # Check competitor 2 keywords not already added
    added_kws = {item["kw"].lower() for item in gaps}
    for kw_item in comp2_keywords:
        kw = kw_item["kw"].lower()
        if kw not in client_kws and kw not in added_kws:
            gaps.append({
                "kw": kw_item["kw"],
                "client_rank": "—",
                "comp1_rank": "—",
                "comp2_rank": kw_item["pos"]
            })
    
    # Sort by competitor position (prioritize keywords where competitors rank well)
    return sorted(gaps, key=lambda x: min(
        x["comp1_rank"] if x["comp1_rank"] != "—" else 100,
        x["comp2_rank"] if x["comp2_rank"] != "—" else 100
    ))[:10]  # Limit to top 10 gaps

def get_trends_data(keywords: List[Dict], timeframe='today 12-m') -> List[Dict]:
    """Get Google Trends data for top keywords to suggest content topics."""
    if not keywords:
        return []
        
    try:
        # Set a short timeout for pytrends
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(3, 8))
        content_ideas = []
        
        # Use top 3 keywords to get related queries
        for kw_item in keywords[:3]:
            kw = kw_item["kw"]
            try:
                pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo='US')
                related_queries = pytrends.related_queries()
                
                if kw in related_queries and 'rising' in related_queries[kw]:
                    rising = related_queries[kw]['rising']
                    if not rising.empty:
                        # Get top rising query
                        top_query = rising.iloc[0]['query']
                        
                        # Create content plan item
                        topic = f"{top_query} {datetime.now().year}"
                        h1 = f"{top_query.title()} in {datetime.now().year}"
                        
                        content_ideas.append({
                            "topic": topic,
                            "intent": "TOFU",  # Top of funnel by default
                            "target_kw": top_query,
                            "h1": h1,
                            "outline": ["Introduction", "Key Benefits", "Case Studies", "FAQ"],
                            "internal_links": [kw_item.get("url", "/")]
                        })
            except Exception as e:
                print(f"[TRENDS] Error getting trends for {kw}: {e}")
                continue
                
        return content_ideas
    except Exception as e:
        print(f"[TRENDS] Google Trends error: {e}")
        return []

def generate_ai_summary(client_data: Dict, comp1_data: Dict, comp2_data: Dict) -> Dict:
    """Generate AI summaries based on actual data."""
    api_key = os.getenv("OPENAI_API_KEY")
    default_response = {
        "executive_summary": "Unable to generate summary. Using data-driven insights to improve your strategy.",
        "competitor_commentary": "Competitor analysis unavailable. Focus on your core strengths and unique value proposition."
    }
    
    if not api_key:
        print("[AI] Missing OPENAI_API_KEY")
        return default_response
    
    try:
        # Prepare data for the AI
        client_tech = client_data.get("sites", {}).get("client", {}).get("tech", {})
        client_keywords = client_data.get("keywords", {}).get("top", [])
        keyword_gaps = client_data.get("keywords", {}).get("gaps", [])
        
        # Format data for the prompt
        tech_summary = (
            f"Core Web Vitals: LCP {client_tech.get('lcp_ms', 0)}ms ({client_tech.get('lcp_status', 'unknown')}), "
            f"CLS {client_tech.get('cls', 0)} ({client_tech.get('cls_status', 'unknown')}), "
            f"INP {client_tech.get('inp_ms', 0)}ms ({client_tech.get('inp_status', 'unknown')}). "
            f"Mobile CWV pass: {client_tech.get('mobile_cwv_pass', 'false')}, "
            f"Desktop CWV pass: {client_tech.get('desktop_cwv_pass', 'false')}."
        )
        
        keyword_summary = "Top keywords: " + ", ".join([
            f"{kw['kw']} (pos {kw['pos']}, vol {kw['vol']})" 
            for kw in client_keywords[:5]
        ]) if client_keywords else "No keyword data available."
        
        gaps_summary = "Keyword gaps: " + ", ".join([
            f"{gap['kw']} (comp1: {gap['comp1_rank']}, comp2: {gap['comp2_rank']})" 
            for gap in keyword_gaps[:5]
        ]) if keyword_gaps else "No keyword gap data available."
        
        # Create the executive summary prompt
        exec_prompt = f"""
        Based on the following real SEO data, write a concise executive summary (max 2 sentences):
        
        {tech_summary}
        
        {keyword_summary}
        
        {gaps_summary}
        
        Focus only on the data provided. Do not invent metrics or insights not present in the data.
        """
        
        # Create the competitor commentary prompt
        comp_prompt = f"""
        Based on the keyword gap analysis below, write a 1-2 sentence competitor commentary:
        
        {gaps_summary}
        
        Focus only on the data provided. Be specific about the competitive landscape based on the keyword gaps.
        """
        
        # Generate executive summary
        client = openai.OpenAI(api_key=api_key)
        exec_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": exec_prompt}],
            max_tokens=150,
            temperature=0.7,
            timeout=10  # 10 second timeout
        )
        
        # Generate competitor commentary
        comp_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": comp_prompt}],
            max_tokens=100,
            temperature=0.7,
            timeout=10  # 10 second timeout
        )
        
        return {
            "executive_summary": exec_response.choices[0].message.content.strip(),
            "competitor_commentary": comp_response.choices[0].message.content.strip()
        }
    except Exception as e:
        print(f"[AI] Summary generation error: {e}")
        return default_response

def generate_quick_wins(tech_data: Dict) -> List[Dict]:
    """Generate quick win recommendations based on Core Web Vitals data."""
    quick_wins = []
    
    # LCP recommendations
    lcp_ms = tech_data.get("lcp_ms", 0)
    if lcp_ms > 2500:
        quick_wins.append({
            "issue": "Slow Largest Contentful Paint",
            "evidence_metric": "LCP",
            "evidence_value": f"{lcp_ms} ms",
            "fix": "Optimize and compress images, implement preloading for critical assets",
            "impact": "high" if lcp_ms > 4000 else "medium",
            "eta_days": 7
        })
    
    # CLS recommendations
    cls = tech_data.get("cls", 0)
    if cls > 0.1:
        quick_wins.append({
            "issue": "Layout shifts affecting user experience",
            "evidence_metric": "CLS",
            "evidence_value": str(cls),
            "fix": "Set explicit dimensions for images and embeds, avoid inserting content above existing content",
            "impact": "high" if cls > 0.25 else "medium",
            "eta_days": 5
        })
    
    # INP recommendations
    inp_ms = tech_data.get("inp_ms", 0)
    if inp_ms > 200:
        quick_wins.append({
            "issue": "Slow interaction responsiveness",
            "evidence_metric": "INP",
            "evidence_value": f"{inp_ms} ms",
            "fix": "Optimize JavaScript execution, implement code splitting, reduce main thread work",
            "impact": "high" if inp_ms > 500 else "medium",
            "eta_days": 10
        })
    
    return quick_wins

def generate_roadmap(tech_data: Dict, keyword_gaps: List[Dict]) -> Dict:
    """Generate a roadmap based on actual data."""
    roadmap = {
        "month_1": [],
        "month_3": [],
        "month_6": []
    }
    
    # Month 1 priorities based on Core Web Vitals
    if tech_data.get("lcp_status") != "pass" or tech_data.get("cls_status") != "pass" or tech_data.get("inp_status") != "pass":
        roadmap["month_1"].append("Fix Core Web Vitals issues")
    
    if keyword_gaps:
        roadmap["month_1"].append("Target top 3 keyword gaps")
    
    # Month 3 priorities
    roadmap["month_3"].append("Publish content for identified keyword gaps")
    roadmap["month_3"].append("Implement internal linking strategy")
    
    # Month 6 priorities
    roadmap["month_6"].append("Scale content production")
    roadmap["month_6"].append("Review and optimize Core Web Vitals")
    
    return roadmap

def generate_backlink_strategy(keyword_gaps: List[Dict]) -> Dict:
    """Generate backlink strategy based on keyword gaps."""
    if not keyword_gaps:
        return {
            "strategy": "Insufficient data to generate backlink strategy",
            "prospects": []
        }
    
    # Extract topics from keyword gaps
    topics = [gap["kw"] for gap in keyword_gaps[:3]]
    
    strategy = f"Target websites ranking for: {', '.join(topics)}"
    
    prospects = [
        {"type": "directory", "why": "Relevant industry listing for targeted keywords"},
        {"type": "guest post", "why": "Build authority in gap areas"}
    ]
    
    return {
        "strategy": strategy,
        "prospects": prospects
    }

def calculate_health_score(tech_data: Dict) -> int:
    """Calculate a health score based on Core Web Vitals."""
    score = 0
    
    # LCP scoring (max 33 points)
    lcp_ms = tech_data.get("lcp_ms", 0)
    if lcp_ms <= 2500:
        score += 33
    elif lcp_ms <= 4000:
        score += 20
    else:
        score += 5
    
    # CLS scoring (max 33 points)
    cls = tech_data.get("cls", 0)
    if cls <= 0.1:
        score += 33
    elif cls <= 0.25:
        score += 20
    else:
        score += 5
    
    # INP scoring (max 34 points)
    inp_ms = tech_data.get("inp_ms", 0)
    if inp_ms <= 200:
        score += 34
    elif inp_ms <= 500:
        score += 20
    else:
        score += 5
    
    return score

def estimate_domain_metrics(keywords: List[Dict]) -> Dict:
    """Estimate domain metrics based on keyword data."""
    if not keywords:
        return {
            "health": "—",
            "da": "—",
            "traffic": "—",
            "kw": "—"
        }
    
    # Count keywords
    kw_count = len(keywords)
    
    # Estimate traffic based on positions and volumes
    estimated_traffic = 0
    for kw in keywords:
        pos = kw.get("pos", 0)
        vol = kw.get("vol", 0)
        
        # CTR estimation based on position
        if pos == 1:
            ctr = 0.3  # 30% CTR for position 1
        elif pos <= 3:
            ctr = 0.15  # 15% CTR for positions 2-3
        elif pos <= 5:
            ctr = 0.08  # 8% CTR for positions 4-5
        elif pos <= 10:
            ctr = 0.03  # 3% CTR for positions 6-10
        else:
            ctr = 0.01  # 1% CTR for positions beyond 10
        
        estimated_traffic += vol * ctr
    
    # Format traffic with k/m suffix
    if estimated_traffic >= 1000000:
        traffic = f"{estimated_traffic/1000000:.1f}m"
    elif estimated_traffic >= 1000:
        traffic = f"{estimated_traffic/1000:.1f}k"
    else:
        traffic = str(int(estimated_traffic))
    
    return {
        "health": f"{min(100, 50 + kw_count)}%",  # Simple health score based on keyword count
        "da": "—",  # We don't have real DA data
        "traffic": traffic,
        "kw": str(kw_count)
    }

def fetch_on_page_issues(url: str) -> List[Dict]:
    """Identify basic on-page issues based on a simple fetch."""
    try:
        session = create_session("ONPAGE", 5, 8)
        resp = session.get(url, timeout=(5, 8))
        html = resp.text.lower()
        
        issues = []
        
        # Check for H1
        if "<h1" not in html:
            issues.append({
                "url": "/",
                "issue": "Missing H1 tag",
                "fix": "Add a unique H1 tag with target keyword"
            })
        
        # Check for title
        if "<title" not in html:
            issues.append({
                "url": "/",
                "issue": "Missing title tag",
                "fix": "Add a descriptive title with primary keyword"
            })
        
        # Check for meta description
        if "meta name=\"description\"" not in html:
            issues.append({
                "url": "/",
                "issue": "Missing meta description",
                "fix": "Add a compelling meta description with primary keyword"
            })
        
        return issues
    except Exception as e:
        print(f"[ONPAGE] Error fetching on-page issues: {e}")
        return [{
            "url": "/",
            "issue": f"Error analyzing page: {str(e)[:100]}",
            "fix": "Check site accessibility"
        }]

def generate_rewrites(url: str, keywords: List[Dict]) -> List[Dict]:
    """Generate title and meta description rewrites based on keywords."""
    if not keywords:
        return []
    
    # Extract domain for title
    domain = urlparse(url).netloc
    if not domain:
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
    
    # Use top keyword for rewrite
    top_kw = keywords[0]["kw"] if keywords else "Your Main Service"
    
    return [{
        "url": "/",
        "example_title": f"{top_kw.title()} | {domain}",
        "example_meta": f"Discover {top_kw} solutions that deliver results. Expert {top_kw.lower()} services tailored to your needs."
    }]

# ---------------------------
# Data Assembly
# ---------------------------
def create_base_payload(client_url: str) -> Dict:
    """Create a base payload with default structure."""
    # Extract domain name for client name
    domain = urlparse(client_url).netloc
    if not domain:
        domain = client_url.replace("https://", "").replace("http://", "").split("/")[0]
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    return {
        "client_name": domain,
        "your_logo_url": "https://via.placeholder.com/140x40?text=nykra",
        "client_logo_url": f"https://logo.clearbit.com/{domain}",
        "report_date": current_date,
        "year": str(current_year),
        "your_company": "nykra studio",
        "cta_url": "https://nykra.studio/book",
        "contact_email": "hello@nykra.studio",
        
        "kpi": {"da": "—", "organic_visits": "—", "keyword_count": "—", "health_score": "—"},
        
        "sites": {
            "client": {
                "tech": {}
            }
        },
        
        "strategy": {
            "executive_summary": "",
            "quick_wins": [],
            "content_plan": [],
            "backlinks": {"strategy": "", "prospects": []},
            "roadmap": {"month_1": [], "month_3": [], "month_6": []},
            "competitor_commentary": ""
        },
        
        "keywords": {
            "top": [],
            "gaps": []
        },
        
        "on_page": {
            "issues": [],
            "rewrites": []
        },
        
        "backlinks": {
            "ref_domains": 0,
            "total": 0,
            "top_domains": []
        },
        
        "bench": {
            "client": {"health": "—", "da": "—", "traffic": "—", "kw": "—"},
            "comp1": {"health": "—", "da": "—", "traffic": "—", "kw": "—"},
            "comp2": {"health": "—", "da": "—", "traffic": "—", "kw": "—"}
        }
    }

def fetch_all_data(client_url: str, comp1_url: Optional[str] = None, comp2_url: Optional[str] = None) -> Dict:
    """Fetch all data from APIs and assemble the complete report payload."""
    # Initialize with base structure
    data = create_base_payload(client_url)
    
    # Ensure URLs have protocol
    if not urlparse(client_url).scheme:
        client_url = "https://" + client_url
    
    if comp1_url and not urlparse(comp1_url).scheme:
        comp1_url = "https://" + comp1_url
    
    if comp2_url and not urlparse(comp2_url).scheme:
        comp2_url = "https://" + comp2_url
    
    # 1. Fetch Core Web Vitals in parallel
    psi_api_key = os.getenv("GOOGLE_PAGESPEED_API_KEY")
    if psi_api_key:
        try:
            tech_mobile, tech_desktop = psi_fetch_parallel(client_url, psi_api_key)
            
            # Combine mobile and desktop data
            data["sites"]["client"]["tech"] = {
                **tech_mobile,
                               "mobile_cwv_pass": "true" if all(
                    tech_mobile[k] == "pass" for k in ("lcp_status", "inp_status", "cls_status")
                ) else "false",
                "desktop_cwv_pass": "true" if all(
                    tech_desktop[k] == "pass" for k in ("lcp_status", "inp_status", "cls_status")
                ) else "false",
            }
            
            # Calculate health score
            health_score = calculate_health_score(tech_mobile)
            data["kpi"]["health_score"] = str(health_score)
        except Exception as e:
            print(f"[PSI] Error processing PSI data: {e}")
            # Set default tech data on failure
            data["sites"]["client"]["tech"] = {
                "lcp_ms": 0, "lcp_status": "fail",
                "inp_ms": 0, "inp_status": "fail",
                "cls": 0.0, "cls_status": "fail",
                "mobile_cwv_pass": "false",
                "desktop_cwv_pass": "false"
            }
    
    # 2. Fetch keywords data in parallel
    serpstack_api_key = os.getenv("SERPSTACK_API_KEY")
    client_keywords = []
    comp1_keywords = []
    comp2_keywords = []
    
    if serpstack_api_key:
        try:
            client_keywords, comp1_keywords, comp2_keywords = fetch_keywords_parallel(
                client_url, comp1_url, comp2_url, serpstack_api_key
            )
            
            data["keywords"]["top"] = client_keywords
            data["kpi"]["keyword_count"] = str(len(client_keywords))
            
            # Find keyword gaps
            data["keywords"]["gaps"] = get_keyword_gaps(client_keywords, comp1_keywords, comp2_keywords)
        except Exception as e:
            print(f"[SERPSTACK] Error processing keyword data: {e}")
            # Keep empty lists as defaults
    
    # 3. Google Trends data for content plan
    try:
        data["strategy"]["content_plan"] = get_trends_data(client_keywords)
    except Exception as e:
        print(f"[TRENDS] Error generating content plan: {e}")
        data["strategy"]["content_plan"] = []
    
    # 4. On-page issues
    try:
        data["on_page"]["issues"] = fetch_on_page_issues(client_url)
        data["on_page"]["rewrites"] = generate_rewrites(client_url, client_keywords)
    except Exception as e:
        print(f"[ONPAGE] Error processing on-page data: {e}")
        data["on_page"]["issues"] = [{
            "url": "/",
            "issue": "Error analyzing page",
            "fix": "Check site accessibility"
        }]
        data["on_page"]["rewrites"] = []
    
    # 5. Generate quick wins based on Core Web Vitals
    try:
        data["strategy"]["quick_wins"] = generate_quick_wins(data["sites"]["client"]["tech"])
    except Exception as e:
        print(f"[STRATEGY] Error generating quick wins: {e}")
        data["strategy"]["quick_wins"] = []
    
    # 6. Generate backlink strategy
    try:
        data["strategy"]["backlinks"] = generate_backlink_strategy(data["keywords"]["gaps"])
    except Exception as e:
        print(f"[STRATEGY] Error generating backlink strategy: {e}")
        data["strategy"]["backlinks"] = {
            "strategy": "Unable to generate backlink strategy",
            "prospects": []
        }
    
    # 7. Generate roadmap
    try:
        data["strategy"]["roadmap"] = generate_roadmap(
            data["sites"]["client"]["tech"], 
            data["keywords"]["gaps"]
        )
    except Exception as e:
        print(f"[STRATEGY] Error generating roadmap: {e}")
        data["strategy"]["roadmap"] = {
            "month_1": ["Analyze site performance"],
            "month_3": ["Implement content strategy"],
            "month_6": ["Review and optimize"]
        }
    
    # 8. Benchmark data
    try:
        data["bench"]["client"] = estimate_domain_metrics(client_keywords)
        
        if comp1_url:
            data["bench"]["comp1"] = estimate_domain_metrics(comp1_keywords)
        
        if comp2_url:
            data["bench"]["comp2"] = estimate_domain_metrics(comp2_keywords)
    except Exception as e:
        print(f"[BENCH] Error calculating benchmark data: {e}")
        # Keep default benchmark data
    
    # 9. AI-generated summaries
    try:
        ai_summaries = generate_ai_summary(
            data,
            {"keywords": {"top": comp1_keywords}},
            {"keywords": {"top": comp2_keywords}}
        )
        
        data["strategy"]["executive_summary"] = ai_summaries["executive_summary"]
        data["strategy"]["competitor_commentary"] = ai_summaries["competitor_commentary"]
    except Exception as e:
        print(f"[AI] Error generating AI summaries: {e}")
        data["strategy"]["executive_summary"] = "Unable to generate summary. Using data-driven insights to improve your strategy."
        data["strategy"]["competitor_commentary"] = "Competitor analysis unavailable. Focus on your core strengths and unique value proposition."
    
    return data

# ---------------------------
# Template Rendering
# ---------------------------
def render_template_with_data(data: dict) -> str:
    """Safely render template with fallbacks for missing data."""
    if not TEMPLATE_PATH.exists():
        return "Template not found. Create templates/nykra_report.html"
    
    try:
        html_template = TEMPLATE_PATH.read_text(encoding="utf-8")
        # normalize {{#each ...}} … {{/each}} to Mustache sections
        html_template = normalize_handlebars_each_to_mustache(html_template)
        renderer = pystache.Renderer(escape=lambda u: u)  # don't escape; template is trusted
        return renderer.render(html_template, data)
    except Exception as e:
        print(f"[TEMPLATE] Error rendering template: {e}")
        return f"Error rendering template: {str(e)}"

def validate_payload(data: dict):
    """Minimal sanity check; expand later as needed"""
    required = ["client_name", "sites", "strategy"]
    return [k for k in required if k not in data]

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return "SEO service is running!"

@app.route("/diag/psi")
def diag_psi():
    """Quick diagnostic endpoint to check PSI API health."""
    api_key = os.getenv("GOOGLE_PAGESPEED_API_KEY")
    if not api_key:
        return jsonify({"ok": False, "error": "Missing GOOGLE_PAGESPEED_API_KEY"})

    try:
        # Use a very short timeout for quick health check
        resp = PSI_SESSION.get(
            "https://www.googleapis.com/pagespeedonline/v5/runPagespeed",
            params={"url": "https://example.com", "strategy": "mobile", "category": "PERFORMANCE", "key": api_key},
            timeout=(3, 5)
        )
        resp.raise_for_status()
        return jsonify({"ok": True, "status_code": resp.status_code})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/report/psi")
def report_psi():
    """Renders the report with live Core Web Vitals for ?url=... using PSI."""
    api_key = os.getenv("GOOGLE_PAGESPEED_API_KEY")
    if not api_key:
        return "Missing GOOGLE_PAGESPEED_API_KEY. Set this environment variable to use this endpoint.", 400

    raw_url = request.args.get("url", "").strip()
    if not raw_url:
        return "Add ?url=https://example.com", 400
    if not urlparse(raw_url).scheme:
        raw_url = "https://" + raw_url  # ensure scheme

    # Fetch PSI for both strategies in parallel
    try:
        tech_mobile, tech_desktop = psi_fetch_parallel(raw_url, api_key)

        data = create_base_payload(raw_url)
        data["client_name"] = raw_url
        data["sites"]["client"]["tech"] = {
            **tech_mobile,
            "mobile_cwv_pass": "true" if all(
                tech_mobile[k] == "pass" for k in ("lcp_status", "inp_status", "cls_status")
            ) else "false",
            "desktop_cwv_pass": "true" if all(
                tech_desktop[k] == "pass" for k in ("lcp_status", "inp_status", "cls_status")
            ) else "false",
        }

        filled_html = render_template_with_data(data)
        if filled_html.startswith("Template not found"):
            return filled_html, 404
        return Response(filled_html, mimetype="text/html")
    except Exception as e:
        print(f"[REPORT] Error in /report/psi: {e}")
        # Create minimal data with error indicators
        data = create_base_payload(raw_url)
        data["client_name"] = raw_url
        data["sites"]["client"]["tech"] = {
            "lcp_ms": 0, "lcp_status": "fail",
            "inp_ms": 0, "inp_status": "fail",
            "cls": 0.0, "cls_status": "fail",
            "mobile_cwv_pass": "false",
            "desktop_cwv_pass": "false"
        }
        data["strategy"]["executive_summary"] = f"Error fetching PSI data: {str(e)[:100]}"
        
        filled_html = render_template_with_data(data)
        return Response(filled_html, mimetype="text/html")

@app.route("/report/full")
def report_full():
    """Renders a complete report with data from all APIs."""
    # Get URLs from query parameters
    client_url = request.args.get("url", "").strip()
    comp1_url = request.args.get("comp1", "").strip()
    comp2_url = request.args.get("comp2", "").strip()
    
    if not client_url:
        return "Add ?url=example.com&comp1=competitor1.com&comp2=competitor2.com", 400
    
    # Check required API keys
    missing_keys = []
    if not os.getenv("GOOGLE_PAGESPEED_API_KEY"):
        missing_keys.append("GOOGLE_PAGESPEED_API_KEY")
    if not os.getenv("SERPSTACK_API_KEY"):
        missing_keys.append("SERPSTACK_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        # Create a basic report noting the missing keys
        data = create_base_payload(client_url)
        data["strategy"]["executive_summary"] = f"Missing required API keys: {', '.join(missing_keys)}"
        filled_html = render_template_with_data(data)
        return Response(filled_html, mimetype="text/html")
    
    # Fetch all data and render the report
    try:
        data = fetch_all_data(client_url, comp1_url, comp2_url)
        filled_html = render_template_with_data(data)
        
        if filled_html.startswith("Template not found"):
            return filled_html, 404
            
        return Response(filled_html, mimetype="text/html")
    except Exception as e:
        print(f"[REPORT] Error in /report/full: {e}")
        # Create minimal data with error indicators
        data = create_base_payload(client_url)
        data["strategy"]["executive_summary"] = f"Error generating full report: {str(e)[:100]}"
        
        filled_html = render_template_with_data(data)
        return Response(filled_html, mimetype="text/html")

@app.route("/report/render", methods=["POST"])
def report_render():
    """Generic renderer endpoint for n8n - accepts full JSON payload and returns HTML"""
    if not request.is_json:
        return "Send JSON body", 400
    
    try:
        data = request.get_json(silent=True) or {}
        missing = validate_payload(data)
        if missing:
            return jsonify({"error": "missing fields", "fields": missing}), 400

        filled_html = render_template_with_data(data)
        if filled_html.startswith("Template not found"):
            return filled_html, 404
        
        return Response(filled_html, mimetype="text/html")
    except Exception as e:
        print(f"[REPORT] Error in /report/render: {e}")
        return jsonify({"error": f"Error rendering report: {str(e)}"}), 500

@app.route("/api/keywords", methods=["GET"])
def api_keywords():
    """API endpoint to fetch keywords for a domain."""
    domain = request.args.get("domain", "").strip()
    if not domain:
        return jsonify({"ok": False, "error": "Missing domain parameter"}), 400
        
    api_key = os.getenv("SERPSTACK_API_KEY")
    if not api_key:
        return jsonify({"ok": False, "error": "Missing SERPSTACK_API_KEY"}), 400
    
    try:
        keywords = serpstack_fetch_keywords(domain, api_key)
        return jsonify({"ok": True, "domain": domain, "keywords": keywords})
    except Exception as e:
        print(f"[API] Error in /api/keywords: {e}")
        return jsonify({"ok": False, "error": str(e), "domain": domain, "keywords": []}), 200  # Return 200 with error info

@app.route("/api/cwv", methods=["GET"])
def api_cwv():
    """API endpoint to fetch Core Web Vitals for a URL."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "Missing url parameter"}), 400
        
    api_key = os.getenv("GOOGLE_PAGESPEED_API_KEY")
    if not api_key:
        return jsonify({"ok": False, "error": "Missing GOOGLE_PAGESPEED_API_KEY"}), 400
    
    try:
        mobile, desktop = psi_fetch_parallel(url, api_key)
        
        return jsonify({
            "ok": True,
            "url": url,
            "mobile": mobile,
            "desktop": desktop
        })
    except Exception as e:
        print(f"[API] Error in /api/cwv: {e}")
        return jsonify({
            "ok": False,
            "error": str(e),
            "url": url,
            "mobile": {
                "lcp_ms": 0, "lcp_status": "fail",
                "inp_ms": 0, "inp_status": "fail",
                "cls": 0.0, "cls_status": "fail"
            },
            "desktop": {
                "lcp_ms": 0, "lcp_status": "fail",
                "inp_ms": 0, "inp_status": "fail",
                "cls": 0.0, "cls_status": "fail"
            }
        }), 200  # Return 200 with error info

@app.route("/api/trends", methods=["GET"])
def api_trends():
    """API endpoint to fetch Google Trends data for keywords."""
    keywords = request.args.get("keywords", "").strip().split(",")
    if not keywords or keywords[0] == "":
        return jsonify({"ok": False, "error": "Missing keywords parameter"}), 400
    
    try:
        keyword_data = [{"kw": kw} for kw in keywords]
        trends = get_trends_data(keyword_data)
        
        return jsonify({
            "ok": True,
            "keywords": keywords,
            "trends": trends
        })
    except Exception as e:
        print(f"[API] Error in /api/trends: {e}")
        return jsonify({
            "ok": False,
            "error": str(e),
            "keywords": keywords,
            "trends": []
        }), 200  # Return 200 with error info

# Local debug (Render uses gunicorn via Procfile)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
