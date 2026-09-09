"""Configuração central do experimento POLIDriving."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
DATA_FILE = ROOT / 'data/raw/20240208_120000_vld.csv'
REVISION = '47238f96a18cb58733353cb647a0af99d4b4f0b5'
DATA_URL = f'https://raw.githubusercontent.com/laboratorioAI/polidriving/{REVISION}/labeled/20240208_120000_vld.csv'
DATA_SHA256 = 'c57a6e89f07fb7d38eb9055fc3bd7e172f736f8ab251713c5da4a2d663d38eca'
TARGET = 'risk_level'
FEATURES = (
    'observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position',
    'engine_temperature', 'engine_load_value', 'heart_rate', 'current_weather',
    'visibility', 'precipitation', 'accidents_onsite', 'design_speed', 'accidents_time',
)
CATEGORICAL = ('current_weather',)
NUMERIC = tuple(name for name in FEATURES if name not in CATEGORICAL)
TARGET_MAP = {'1': 0, '2': 0, '3': 1, '4': 1,
              'LOW': 0, 'MEDIUM': 0, 'HIGH': 1, 'VERY HIGH': 1}


@dataclass(frozen=True)
class ExperimentConfig:
    data: Path = DATA_FILE
    output: Path = ROOT
    seed: int = RANDOM_STATE
    test_size: float = 0.2
    folds: int = 5
    iterations: int = 20
    jobs: int = 1

    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError('test_size deve estar entre 0 e 1.')
        if self.folds < 2 or self.iterations < 1 or self.jobs == 0:
            raise ValueError('folds >= 2, iterations >= 1 e jobs != 0 são necessários.')
