import pandas as pd
import pytest
from src.config import FEATURES, TARGET


@pytest.fixture
def sample_frame():
    frame = pd.DataFrame({name: [1., 2., 3., 4., 5., 6., 7., 8.] for name in FEATURES})
    frame[TARGET] = [1, 2, 3, 4, 1, 2, 3, 4]
    return frame
