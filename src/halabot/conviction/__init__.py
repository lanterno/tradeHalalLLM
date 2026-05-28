"""conviction — turn a belief's raw evidence signal into a calibrated score.

``conviction_raw`` is the deterministic, LLM-free pre-calibration signal
(REARCHITECTURE B.2). The :class:`Calibrator` maps it to a calibrated
probability of a favorable move; until enough closed outcomes exist to fit one,
:class:`IdentityCalibrator` passes the raw score through (cold-start safe).
"""
