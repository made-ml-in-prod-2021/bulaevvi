from dataclasses import dataclass
from typing import Optional

@dataclass
class ClassifierParams:
    C: Optional[float]
    penalty: Optional[str]
    fit_intercept: Optional[bool]
    random_state: Optional[int]
    n_estimators: Optional[int]
    max_depth: Optional[int]
    random_state: Optional[int]


