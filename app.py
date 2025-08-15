import os, json, pathlib
from flask import Flask, Response, request
import pystache                  # ensure 'pystache' is in requirements.txt
import requests                  # ensure 'requests' is in requirements.txt
from urllib.parse import urlparse

app = Flask(__name__)

TEMPLATE_PATH = pathlib.Path("templates/nykra_report.html").resolve()

# ---------------------------
# Payload used for demo & as a base we overwrite with real data
# ---------------------------
def demo_payload():
    return {
        "client_name": "hellogreenhomes.com",
        "your_logo_url": "https://via.placeholder.com/140x40?text=nykra",
        "client_logo_url": "https://via.placeholder.com/140x40?text=client",
        "report_date": "2025-08-15",
        "year": "2025",
        "your_company": "nykra studio",
        "cta_url": "https://nykra.studio/book",
        "contact_email": "hello@nykra.studio",

        "kpi": {"da": "—", "organic_visits": "—", "keyword_count": "—", "health_score": "—"},

        "sites": {
            "client": {
                "tech": {
                    "lcp_ms": 2350, "lcp_status": "needs improvement",
                    "cls": 0.09, "cls_status": "pass",
                    "inp_ms": 210, "inp_status": "pass",
                    "mobile_cwv_pass": "true", "desktop_cwv_pass": "false"
                }
            }
        },

        "strategy": {
            "executive_summary": "High-level summary based on core metrics and competitor posture.",
            "quick_wins": [
                {"issue":"Compress hero images","evidence_metric":"LCP","evidence_value":"2350 ms","fix":"Serve WebP + preload hero","impact":"high","eta_days":7},
                {"issue":"Thin content on /blog","evidence_metric":"Word count","evidence_value":"<400","fix":"Expand to 1200–1600 words","impact":"med","eta_days":14}
            ],
            "content_plan": [
                {
                    "topic":"Solar-powered homes 2025",
                    "intent":"TOFU",
                    "target_kw":"solar powered homes",
                    "h1":"Solar-Powered Homes in 2025",
                    "outline":["Intro","Benefits","Case studies","FAQ"],
                    "internal_links":["/services/solar","/blog/green-design"]
                }
            ],
            "backlinks":{"strategy":"Target clean-energy directories + partner PR",
                         "prospects":[{"type":"directory","why":"Relevant industry listing"}]},
            "roadmap":{"month_1":["CWV fixes","Top 5 on-page"],
                       "month_3":["Publish 6 posts","Launch outreach"],
                       "month_6":["Scale content","Advanced internal linking"]},
            "competitor_commentary":"Competitors lead on informational depth; publish targeted guides to close gaps."
        },

        "keywords": {
            "top":[{"kw":"solar homes","pos":18,"vol":2900,"url":"/blog/solar-homes"}],
            "gaps":[{"kw":"net zero homes","client_rank":"—","comp1_rank":9,"comp2_rank":12}]
        },

        "on_page":{
            "issues":[{"url":"/","issue":"Missing H1","fix":"Add unique H1 with target keyword"}],
            "rewrites":[{"url":"/","example_title":"Solar Homes For Sale | Cityname","example_meta":"Explore solar-powered homes…"}]
        },

        "backlinks":{
            "ref_domains": 42, "total": 113,
            "top_domains":[{"domain":"example.com","type":"blog","authority":45,"link_type":"follow"}]
        },

        "bench":{
            "client":{"health":"68%","da":"—","traffic":"—","kw":"—"},
            "comp1":{"health":"72%","da":"—","traffic":"—","kw":"—"},
            "comp2":{"health":"65%","da":"—","traffic":"—","kw":"—"}
        }
    }

# ---------------------------
# Helpers
# ---------------------------
def render_template_with_data(data: dict) -> str:
    if not TEMPLATE_PATH.exists():
        return "Template not found. Create templates/nykra_report.html"
    html_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    renderer = pystache.Renderer(escape=lambda u: u)  # don't escape; template is trusted
    return renderer.render(html_template, data)

def psi_fetch(url: str, strategy: str, api_key: str) -> dict:
    """Fetch Core Web Vitals from Google PageSpeed Insights (robust INP extraction)."""
    resp = requests.get(
        "https://www.googleapis.com/pagespeedonline/v5/runPagespeed",
        params={"url": url, "strategy": strategy, "category": "PERFORMANCE", "key": api_key},
        timeout=60
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

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return "SEO service is running!"

@app.route("/report/demo")
def report_demo():
    filled_html = render_template_with_data(demo_payload())
    if filled_html.startswith("Template not found"):
        return filled_html, 500
    return Response(filled_html, mimetype="text/html")

@app.route("/report/psi")
def report_psi():
    """Renders the report with live Core Web Vitals for ?url=... using PSI."""
    api_key = os.getenv("GOOGLE_PAGESPEED_API_KEY")
    if not api_key:
        return "Missing GOOGLE_PAGESPEED_API_KEY", 500

    raw_url = request.args.get("url", "").strip()
    if not raw_url:
        return "Add ?url=https://example.com", 400
    if not urlparse(raw_url).scheme:
        raw_url = "https://" + raw_url  # ensure scheme

    # Fetch PSI for both strategies
    tech_mobile = psi_fetch(raw_url, "mobile", api_key)
    tech_desktop = psi_fetch(raw_url, "desktop", api_key)

    data = demo_payload()
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
        return filled_html, 500
    return Response(filled_html, mimetype="text/html")

# Local debug (Render uses gunicorn via Procfile)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
