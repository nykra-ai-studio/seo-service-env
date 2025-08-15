import os, json, pathlib
from flask import Flask, Response
import pystache  # make sure 'pystache' is in requirements.txt

app = Flask(__name__)

TEMPLATE_PATH = pathlib.Path("templates/nykra_report.html").resolve()

def demo_payload():
    # Minimal but realistic payload to light up all sections
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

@app.route("/")
def home():
    return "SEO service is running!"

@app.route("/report/demo")
def report_demo():
    if not TEMPLATE_PATH.exists():
        return "Template not found. Create templates/nykra_report.html", 500
    html_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # Do not escape because template contains HTML and we trust our own placeholders
    renderer = pystache.Renderer(escape=lambda u: u)
    filled_html = renderer.render(html_template, demo_payload())
    return Response(filled_html, mimetype="text/html")

# Local debug (Render uses gunicorn via Procfile)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
