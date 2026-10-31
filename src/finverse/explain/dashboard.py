"""Interactive dashboard: one self-contained HTML file (no internet, no install). Pick a decision date with the slider.
All numbers come from the logged run; nothing is computed in the browser except cumulative growth."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .writer import TemplateWriter


def _date_payload(d, st, target, ap):
    wts = st["agent_weights"]
    names = [a for a in wts]
    held = target[target > 0.005].sort_values(ascending=False)
    wv = np.array([wts[a] for a in names])
    hs, hold_info = [], []
    for s, w in held.items():
        sc = {a: float(ap.scores[a].loc[d, s]) for a in names if np.isfinite(ap.scores[a].loc[d, s])}
        if not sc:
            continue
        sv = np.array([sc.get(a, np.nan) for a in names]); ok = np.isfinite(sv)
        mean = float((wv[ok] * sv[ok]).sum() / wv[ok].sum())
        dis = float(np.sqrt((wv[ok] * (sv[ok] - mean) ** 2).sum() / wv[ok].sum()))
        c = st["combined"].get(s, np.nan)
        hs.append({"stock": s, "weight": float(w), "combined": float(c) if np.isfinite(c) else None, "scores": sc})
        hold_info.append({"stock": s, "weight": float(w), "combined_score": float(c) if np.isfinite(c) else 0.0,
                          "agent_scores": sc, "disagreement": dis, "counterfactual_weight_change_if_agent_removed": {}})
    e = {"date": str(d.date()), "regime_prob": st["regime_prob"], "agent_weights": wts, "holdings": hold_info}
    return {"date": str(d.date()), "regime_prob": st["regime_prob"], "agent_weights": wts,
            "reason": st["info"].get("reason", ""), "holdings": hs, "text": TemplateWriter().write(e),
            "cash": float(1 - target.sum())}


def build_dashboard(path, title, results, ap, notes, strategies=("dynamic", "equal", "equal_weight", "min_variance")):
    dyn = results["dynamic"]
    dates = [d for d in dyn.state]
    per = [_date_payload(d, dyn.state[d], dyn.weights.loc[d], ap) for d in dates]
    growth = {}
    for s in strategies:
        if s in results:
            r = results[s].returns["net"].reindex(dates).fillna(0.0)
            growth[s] = [float(x) for x in (1 + r).cumprod()]
    data = {"title": title, "dates": [str(d.date()) for d in dates], "per": per, "growth": growth, "notes": notes}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, default=float))
    open(path, "w", encoding="utf-8").write(html)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinVerse dashboard</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;--a:#2a6f97;--b:#c1666b;--c:#6a994e;--d:#8e7dbe}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9c9b94;--card:#1f1f1d;--line:#34332f;--a:#6fb1d6;--b:#e08a8f;--c:#9bc57f;--d:#b5a7e0}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}h1{font-size:20px;margin:8px 0}h2{font-size:15px;margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.bar{height:14px;border-radius:3px;background:var(--a)}.row{display:flex;align-items:center;gap:8px;margin:4px 0}.row span.l{width:150px;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.mut{color:var(--mut);font-size:13px}input[type=range]{width:100%}pre{white-space:pre-wrap;font:13px/1.5 inherit;margin:0}
.warn{border-left:4px solid var(--b);padding-left:10px}</style></head><body><main>
<h1 id="t"></h1><div class="card warn mut" id="notes"></div>
<div class="card"><h2>Decision date: <span id="dd"></span></h2><input type="range" id="sl" min="0" value="0"><div class="mut">Decisions are made at the close of this date and traded at the next close.</div></div>
<div class="grid"><div class="card"><h2>Market state (filtered probabilities, no look-ahead)</h2><div id="reg"></div></div>
<div class="card"><h2>Trust in each model (from past realized results)</h2><div id="aw"></div></div></div>
<div class="card"><h2>Growth of 1 unit since the first decision (net of costs; cost level is a placeholder)</h2><svg id="ch" viewBox="0 0 900 260" width="100%"></svg><div id="lg" class="mut"></div></div>
<div class="card"><h2>Holdings and each model's score (0 = worst rank, 1 = best)</h2><div id="ho"></div></div>
<div class="card"><h2>Plain-English explanation</h2><pre id="tx"></pre></div></main>
<script>
const D=__DATA__;const $=i=>document.getElementById(i);const col=['var(--a)','var(--b)','var(--c)','var(--d)'];
$('t').textContent=D.title;$('notes').innerHTML=D.notes.map(n=>'<div>'+n+'</div>').join('');
const sl=$('sl');sl.max=D.dates.length-1;sl.value=D.dates.length-1;
function bars(o,max){return Object.entries(o).sort((a,b)=>b[1]-a[1]).map(([k,v])=>'<div class="row"><span class="l">'+k.replace(/_/g,' ')+'</span><div class="bar" style="width:'+Math.max(2,300*v/max)+'px"></div><span>'+(100*v).toFixed(0)+'%</span></div>').join('')}
function chart(i){const names=Object.keys(D.growth);let all=[].concat(...names.map(n=>D.growth[n]));const lo=Math.min(...all),hi=Math.max(...all),n=D.dates.length;
const X=j=>40+820*j/(n-1),Y=v=>230-210*(v-lo)/(hi-lo);let s='';
for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;s+='<line x1="40" x2="860" y1="'+Y(v)+'" y2="'+Y(v)+'" stroke="var(--line)"/><text x="4" y="'+(Y(v)+4)+'" font-size="11" fill="var(--mut)">'+v.toFixed(2)+'</text>'}
names.forEach((nm,k)=>{s+='<polyline fill="none" stroke="'+col[k%4]+'" stroke-width="'+(nm=='dynamic'?2.5:1.5)+'" points="'+D.growth[nm].map((v,j)=>X(j)+','+Y(v)).join(' ')+'"/>'});
s+='<line x1="'+X(i)+'" x2="'+X(i)+'" y1="10" y2="230" stroke="var(--fg)" stroke-dasharray="4"/>';
s+='<text x="40" y="252" font-size="11" fill="var(--mut)">'+D.dates[0]+'</text><text x="780" y="252" font-size="11" fill="var(--mut)">'+D.dates[n-1]+'</text>';
$('ch').innerHTML=s;$('lg').innerHTML=names.map((nm,k)=>'<span style="color:'+col[k%4]+'">&#9632;</span> '+nm.replace(/_/g,' ')+' '+D.growth[nm][i].toFixed(2)).join(' &nbsp; ')}
function show(i){const p=D.per[i];$('dd').textContent=p.date;$('reg').innerHTML=bars(p.regime_prob,1);$('aw').innerHTML=bars(p.agent_weights,1);
const ag=Object.keys(p.agent_weights);
$('ho').innerHTML='<table><tr><th>Stock</th><th>Weight</th><th>Combined</th>'+ag.map(a=>'<th>'+a.replace(/_/g,' ')+'</th>').join('')+'</tr>'+
p.holdings.map(h=>'<tr><td>'+h.stock.replace(/_/g,' ')+'</td><td>'+(100*h.weight).toFixed(1)+'%</td><td>'+(h.combined==null?'-':h.combined.toFixed(2))+'</td>'+ag.map(a=>'<td>'+(h.scores[a]==null?'-':h.scores[a].toFixed(2))+'</td>').join('')+'</tr>').join('')+'</table>'+
'<div class="mut">Cash '+(100*p.cash).toFixed(1)+'%. Only holdings above 0.5% shown.</div>';
$('tx').textContent=p.text;chart(i)}
sl.oninput=()=>show(+sl.value);show(+sl.value);
</script></body></html>"""
