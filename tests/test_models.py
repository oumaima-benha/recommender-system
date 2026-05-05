import pytest
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.hybrid import hybrid_recommendation


def test_hybrid_output_type():
    recs = hybrid_recommendation(user_id=1, top_n=5)
    
    assert isinstance(recs, list)


def test_hybrid_length():
    recs = hybrid_recommendation(user_id=1, top_n=5)
    
    assert len(recs) <= 5


def test_hybrid_not_empty():
    recs = hybrid_recommendation(user_id=1, top_n=5)
    
    assert len(recs) > 0


def test_unknown_user():
    recs = hybrid_recommendation(user_id=999999, top_n=5)
    
    # dépend de ton implémentation
    assert isinstance(recs, list)