"""HTML/CSS helpers for the dashboard (pure functions, no Streamlit import)."""
import math

TIER_COLORS = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"}
TIER_MESSAGES = {
    "Low": "No early-warning alert. Keep up your current habits and check in again if things change.",
    "Medium": "Early-warning alert. Consider a check-in with a counselor before things escalate.",
    "High": "High risk. We strongly recommend talking with a qualified counselor soon.",
}

CSS = """
<style>
.block-container{max-width:960px;padding-top:4rem}
.hero{background:linear-gradient(135deg,#4f46e5 0%,#0ea5e9 60%,#14b8a6 100%);color:#fff;
  padding:2rem 2rem 1.6rem;border-radius:20px;margin-bottom:1rem;box-shadow:0 8px 24px rgba(79,70,229,.25)}
.hero h1{color:#fff;margin:0 0 .4rem;font-size:2rem;line-height:1.2;padding:0}
.hero p{margin:0;opacity:.95;font-size:1.02rem}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.9rem 0 0}
.chip{background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);padding:.25rem .7rem;border-radius:999px;font-size:.82rem}
.card{border:1px solid rgba(128,128,128,.25);background:rgba(128,128,128,.06);border-radius:16px;padding:1.1rem 1.2rem}
.badge{display:inline-block;color:#fff;font-weight:700;padding:.3rem .9rem;border-radius:999px;font-size:1.05rem;letter-spacing:.02em}
.banner{border-left:6px solid;border-radius:10px;padding:.8rem 1rem;margin-top:.8rem;font-weight:500}
.frow{display:flex;align-items:center;gap:.6rem;margin:.35rem 0;font-size:.92rem}
.flabel{flex:0 0 46%}
.ftrack{flex:1;height:10px;background:rgba(128,128,128,.18);border-radius:6px;overflow:hidden}
.fbar{height:100%;border-radius:6px}
.small{font-size:.82rem;opacity:.75}
</style>
"""


def hero_html():
    return (
        '<div class="hero"><h1>🧠 AI-Based Early Burnout Detection</h1>'
        "<p>A two-minute check-in that estimates your risk from study, sleep and stress "
        "signals, then suggests practical next steps.</p>"
        '<div class="chips"><span class="chip">⏱ ~2 minutes</span>'
        '<span class="chip">🔒 Answers are not stored by this app</span>'
        '<span class="chip">🧪 Research prototype, not a diagnosis</span></div></div>'
    )


def gauge_svg(prob: float, threshold: float) -> str:
    """Semicircular gauge: green/amber/red zones split at the alert threshold and 65%."""
    L = math.pi * 90  # arc length of the r=90 half circle
    t_hi = max(0.65, threshold)
    seg = lambda s, e, c: (  # noqa: E731
        f'<path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="{c}" stroke-width="16" '
        f'stroke-dasharray="{(e - s) * L:.2f} {L:.2f}" stroke-dashoffset="{-s * L:.2f}" opacity=".85"/>'
    )
    a = math.pi * (1 - min(max(prob, 0), 1))
    x, y = 100 + 74 * math.cos(a), 100 - 74 * math.sin(a)
    return (
        '<svg viewBox="0 0 200 134" style="width:100%;max-width:340px;display:block;margin:0 auto">'
        + seg(0, threshold, TIER_COLORS["Low"]) + seg(threshold, t_hi, TIER_COLORS["Medium"])
        + seg(t_hi, 1, TIER_COLORS["High"])
        + f'<line x1="100" y1="100" x2="{x:.1f}" y2="{y:.1f}" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>'
        '<circle cx="100" cy="100" r="7" fill="currentColor"/>'
        f'<text x="100" y="128" text-anchor="middle" font-size="20" font-weight="700" fill="currentColor">{prob * 100:.0f}%</text>'
        "</svg>"
    )


def tier_html(tier: str) -> str:
    c = TIER_COLORS[tier]
    return (
        f'<span class="badge" data-tier="{tier}" style="background:{c}">{tier} risk</span>'
        f'<div class="banner" data-tier="{tier}" style="border-color:{c};background:{c}1a">{TIER_MESSAGES[tier]}</div>'
    )


def factor_rows_html(items, color: str) -> str:
    """items: [(label, contribution)] -> horizontal bars scaled to the largest |value|."""
    if not items:
        return '<div class="small">Nothing notable.</div>'
    top = max(abs(v) for _, v in items) or 1.0
    return "".join(
        f'<div class="frow"><div class="flabel">{label}</div><div class="ftrack">'
        f'<div class="fbar" style="width:{max(abs(v) / top * 100, 4):.0f}%;background:{color}"></div></div></div>'
        for label, v in items
    )
