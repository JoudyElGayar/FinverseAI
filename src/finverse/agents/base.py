from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class AgentOutput:
    agent: str
    date: pd.Timestamp
    score: pd.Series        # 0-1, cross-sectional rank percentile, index = stocks (NaN if not scored)
    confidence: pd.Series   # 0-1, relative
    source_ids: list = field(default_factory=list)


class Agent(ABC):
    name: str = "agent"
    is_standin: bool = False

    def fit(self, ds, train_dates) -> None:  # rule agents have nothing to fit
        return None

    @abstractmethod
    def score(self, ds, date) -> AgentOutput:
        ...

    def drivers(self, ds, date, stock, top=3) -> list:
        return []
