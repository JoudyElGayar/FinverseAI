"""Stand-in rule agents. Used only to exercise the coordinator until Fundamental / News / Macro agents exist."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Agent, AgentOutput


class MomentumRuleAgent(Agent):
    name = "momentum_rule"
    is_standin = True

    def score(self, ds, date):
        s = pd.concat([ds.feats["mom_12_1"].loc[date], ds.feats["mom_6m"].loc[date]], axis=1).mean(axis=1)
        s = s.where(ds.elig.loc[date]).rank(pct=True)
        return AgentOutput(self.name, date, s, pd.Series(1.0, index=s.index).where(s.notna()))

    def drivers(self, ds, date, stock, top=3):
        return [{"feature": "mom_12_1", "effect": None, "rank_value": float(ds.feats["mom_12_1"].loc[date, stock])},
                {"feature": "mom_6m", "effect": None, "rank_value": float(ds.feats["mom_6m"].loc[date, stock])}][:top]


class LowVolRuleAgent(Agent):
    name = "lowvol_rule"
    is_standin = True

    def score(self, ds, date):
        s = (1.0 - ds.feats["vol_3m"].loc[date]).where(ds.elig.loc[date]).rank(pct=True)
        return AgentOutput(self.name, date, s, pd.Series(1.0, index=s.index).where(s.notna()))

    def drivers(self, ds, date, stock, top=3):
        return [{"feature": "vol_3m", "effect": None, "rank_value": float(ds.feats["vol_3m"].loc[date, stock])}]


REGISTRY = {"momentum_rule": MomentumRuleAgent, "lowvol_rule": LowVolRuleAgent}
