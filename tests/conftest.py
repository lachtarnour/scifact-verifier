"""Shared fixtures for all tests."""

import pytest


@pytest.fixture
def tiny_corpus():
    return {
        "doc1": {
            "title": "Statins and atrial fibrillation",
            "text": (
                "Statins significantly reduce the incidence of atrial fibrillation "
                "following coronary artery bypass grafting surgery. "
                "This effect is mediated through anti-inflammatory mechanisms."
            ),
        },
        "doc2": {
            "title": "Vitamin D and respiratory infections",
            "text": (
                "Vitamin D supplementation reduces the risk of acute respiratory tract "
                "infections. The protective effect is strongest in individuals with "
                "baseline vitamin D deficiency."
            ),
        },
        "doc3": {
            "title": "BRCA2 and ovarian cancer",
            "text": (
                "BRCA2 germline mutations are associated with a significantly elevated "
                "lifetime risk of ovarian cancer, estimated at 11–30 percent."
            ),
        },
    }


@pytest.fixture
def tiny_qrels():
    return {
        "q1": {"doc1": 1},
        "q2": {"doc2": 1},
        "q3": {"doc3": 1},
    }


@pytest.fixture
def tiny_results():
    return {
        "q1": {"doc1": 0.95, "doc2": 0.4, "doc3": 0.1},
        "q2": {"doc2": 0.88, "doc1": 0.3, "doc3": 0.2},
        "q3": {"doc3": 0.75, "doc2": 0.5, "doc1": 0.1},
    }
