from dataclasses import dataclass

@dataclass
class SplittingParams:
    validation_size: float = 0.2
    random_state: int = 42