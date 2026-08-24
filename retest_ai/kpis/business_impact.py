import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_time_and_yield_impact(
    df: pd.DataFrame,
    recommendations_col: str = "Recommendation",
    ground_truth_col: str = "Ground_Truth",
    retest_time_col: str = "Retest_Time_sec"
) -> Dict[str, Any]:
    """
    Computes business and operational impact in ATE test seconds and device counts.
    Strictly avoids fabricated financial conversions.
    """
    total_events = len(df)
    has_gt = ground_truth_col in df.columns
    has_time = retest_time_col in df.columns

    retest_recs = (df[recommendations_col] == "RETEST")
    skip_recs = (df[recommendations_col] == "DON'T RETEST")

    total_retests_recommended = int(retest_recs.sum())
    total_skips_recommended = int(skip_recs.sum())

    total_actual_retest_time = float(df[retest_time_col].sum()) if has_time else 0.0
    recommended_retest_time = float(df.loc[retest_recs, retest_time_col].sum()) if has_time else 0.0

    unnecessary_retests_count = 0
    unnecessary_retest_time_sec = 0.0
    missed_recoverable_devices = 0
    correctly_recovered_devices = 0

    if has_gt:
        gt_beneficial = (df[ground_truth_col] == "RETEST_BENEFICIAL")
        gt_persistent = (df[ground_truth_col] == "PERSISTENT_FAILURE")

        fp_mask = retest_recs & gt_persistent
        tp_mask = retest_recs & gt_beneficial
        fn_mask = skip_recs & gt_beneficial
        tn_mask = skip_recs & gt_persistent

        unnecessary_retests_count = int(fp_mask.sum())
        correctly_recovered_devices = int(tp_mask.sum())
        missed_recoverable_devices = int(fn_mask.sum())

        if has_time:
            unnecessary_retest_time_sec = float(df.loc[fp_mask, retest_time_col].sum())

    return {
        "total_events": total_events,
        "retest_recommendations_count": total_retests_recommended,
        "retest_recommendations_pct": (total_retests_recommended / total_events * 100.0) if total_events > 0 else 0.0,
        "skip_recommendations_count": total_skips_recommended,
        "skip_recommendations_pct": (total_skips_recommended / total_events * 100.0) if total_events > 0 else 0.0,
        "total_retest_time_sec": total_actual_retest_time,
        "ai_recommended_retest_time_sec": recommended_retest_time,
        "unnecessary_retests_count": unnecessary_retests_count,
        "unnecessary_retest_time_sec": unnecessary_retest_time_sec,
        "correctly_recovered_devices_count": correctly_recovered_devices,
        "missed_recoverable_devices_count": missed_recoverable_devices
    }
