"""Plain-English explanation writers. Every number in the text comes from the logged explanation dict.
TemplateWriter needs no LLM. LLMWriter is a pluggable stub: an LLM may only rephrase the template text and
its output is rejected if it contains a number that is not in the logged facts (see check_numbers)."""
from __future__ import annotations
import re

NAMES = {"technical_ml": "the machine-learning model", "fundamental": "the fundamentals model", "fundamental_rule": "the fundamentals model",
         "momentum_rule": "the momentum rule", "lowvol_rule": "the low-volatility rule", "news_sentiment": "the news model"}


def _n(a):
    return NAMES.get(a, a.replace("_", " "))


def _strength(x):
    return "strongly" if abs(x - 0.5) > 0.3 else "moderately" if abs(x - 0.5) > 0.15 else "only slightly"


class TemplateWriter:
    def write(self, e: dict) -> str:
        top = max(e["regime_prob"], key=e["regime_prob"].get)
        out = [f"Decision date: {e['date']}. The market-state model puts the highest probability "
               f"({e['regime_prob'][top]:.0%}) on {top}."]
        w = sorted(e["agent_weights"].items(), key=lambda kv: -kv[1])
        out.append("Trust given to each model (from how well its past scores matched realized results): "
                   + "; ".join(f"{_n(a)} {v:.0%}" for a, v in w) + ".")
        for h in e["holdings"]:
            sc = h["agent_scores"]
            hi = max(sc, key=sc.get); lo = min(sc, key=sc.get)
            p = [f"{h['stock']} is held at {h['weight']:.1%}. Combined score {h['combined_score']:.2f} (0 = worst, 1 = best rank among eligible stocks)."]
            p.append(f"Most positive: {_n(hi)} ({sc[hi]:.2f}, {_strength(sc[hi])} positive). "
                     f"Most negative: {_n(lo)} ({sc[lo]:.2f}).")
            if h["disagreement"] > 0.2:
                p.append(f"The models disagree here (spread {h['disagreement']:.2f}), so this holding is less certain.")
            cf = h["counterfactual_weight_change_if_agent_removed"]
            if cf:
                k = max(cf, key=lambda a: abs(cf[a]))
                p.append(f"Removing {_n(k)} would change the weight by {cf[k]:+.1%}.")
            out.append(" ".join(p))
        return "\n\n".join(out)


def _numbers(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def check_numbers(llm_text: str, template_text: str) -> bool:
    """True only if every number in the LLM text also appears in the template text."""
    return _numbers(llm_text) <= _numbers(template_text)


class LLMWriter:
    """complete(prompt)->str is supplied by the caller (Claude API client or a local model).
    Falls back to the template text if the LLM invents a number. Not run yet: no API key."""
    def __init__(self, complete):
        self.complete, self.base = complete, TemplateWriter()

    def write(self, e):
        t = self.base.write(e)
        try:
            r = self.complete("Rewrite this for a non-expert investor. Keep every number exactly; add none; "
                              "do not give advice.\n\n" + t)
        except Exception:
            return t
        return r if check_numbers(r, t) else t
