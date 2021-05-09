from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureParams:
    numerical_features: List[str]
    target_col: Optional[str]
    transform_type: str
