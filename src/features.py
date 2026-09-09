"""Seleção explícita: índice do CSV e target nunca entram nos preditores."""
from .config import TARGET
from .data_loader import transform_target, validate_features


def split_features_target(frame):
    return validate_features(frame), transform_target(frame[TARGET])
