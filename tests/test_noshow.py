from datetime import datetime

from domains.clinic.noshow import NoShowModel, features


def test_features_without_history():
    x = features("Терапевт", datetime(2026, 10, 19, 8, 0), None)
    assert x == {"spec:терапевт": 1.0, "weekday:0": 1.0, "hour:early": 1.0}
    x = features("Терапевт", datetime(2026, 10, 19, 12, 0), 4, 2)
    assert x["noshow_rate"] == 0.5 and x["first_visit"] == 0.0


def test_risk_and_explanation():
    m = NoShowModel({"bias": -2.0, "spec:оториноларинголог": 0.3, "hour:early": 0.2, "noshow_rate": 0.9})
    low = m.risk("терапевт", datetime(2026, 10, 19, 12, 0))
    high = m.risk("лор (отоларинголог)", datetime(2026, 10, 19, 8, 0))  # специальность стенда сопоставлена
    assert 0 < low < high < 1
    assert m.explain("оториноларинголог", datetime(2026, 10, 19, 8, 0), 2, 2)[0] == "часто не приходит"
