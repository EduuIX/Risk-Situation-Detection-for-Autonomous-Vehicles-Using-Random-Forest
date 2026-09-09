"""Download verificável e validação estrita do CSV público."""
import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_FILE, DATA_URL, DATA_SHA256, FEATURES, TARGET, TARGET_MAP
from .utils import digest


def transform_target(series: pd.Series) -> pd.Series:
    """Converte os quatro níveis originais; não interpreta 0/1 como níveis originais."""
    labels = series.astype('string').str.strip().str.upper().str.replace('_', ' ', regex=False)
    labels = labels.str.replace(r'^([1-4])\.0+$', r'\1', regex=True)
    invalid = labels.isna() | ~labels.isin(TARGET_MAP)
    if invalid.any():
        raise ValueError(f'Target ausente ou desconhecido: {series[invalid].unique().tolist()}')
    return labels.map(TARGET_MAP).astype(int).rename('risk')


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f'Features obrigatórias ausentes: {missing}')
    result = frame.loc[:, list(FEATURES)].copy()
    for name in FEATURES:
        try:
            result[name] = pd.to_numeric(result[name], errors='raise').astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(f'Feature {name} contém valores não numéricos.') from error
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Features contêm valores infinitos.')
    # Remove somente ruído de representação decimal dos códigos meteorológicos.
    result['current_weather'] = result['current_weather'].round(6)
    return result


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset não encontrado em {path}. Execute python -m src.data_loader ou use --data CAMINHO.')
    frame = pd.read_csv(path)
    if TARGET not in frame or frame.empty:
        raise ValueError(f'O CSV deve conter registros e o target {TARGET}.')
    extra = set(frame) - set(FEATURES) - {TARGET, 'Unnamed: 0'}
    if extra:
        raise ValueError(f'Colunas não previstas no esquema: {sorted(extra)}')
    result = validate_features(frame)
    transform_target(frame[TARGET])
    result[TARGET] = frame[TARGET]
    return result


def inspect_data(frame: pd.DataFrame) -> dict:
    buffer = io.StringIO()
    frame.info(buf=buffer)
    y = transform_target(frame[TARGET])
    return {
        'rows': len(frame), 'features': len(FEATURES), 'shape': list(frame.shape),
        'head': frame.head().to_dict(orient='records'), 'tail': frame.tail().to_dict(orient='records'),
        'info': buffer.getvalue(), 'describe': frame.describe(include='all').to_dict(),
        'dtypes': frame.dtypes.astype(str).to_dict(),
        'missing_count': frame.isna().sum().to_dict(),
        'missing_percentage': frame.isna().mean().mul(100).to_dict(),
        'duplicates': int(frame.duplicated().sum()),
        'original_classes': frame[TARGET].value_counts().to_dict(),
        'binary_classes': y.value_counts().sort_index().to_dict(),
        'binary_percentages': y.value_counts(normalize=True).sort_index().mul(100).to_dict(),
    }


def download_data(destination: Path = DATA_FILE) -> Path:
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != DATA_SHA256:
            raise ValueError('Arquivo existente difere da fonte fixada; não será sobrescrito.')
        return destination
    payload = urllib.request.urlopen(DATA_URL, timeout=60).read()
    import hashlib
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError('Checksum do download inválido.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA_FILE)
    args = parser.parse_args()
    print(download_data(args.output))
