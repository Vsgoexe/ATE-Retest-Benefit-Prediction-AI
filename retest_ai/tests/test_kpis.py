import unittest
import pandas as pd
from retest_ai.kpis.decision_quality import calculate_decision_quality
from retest_ai.validation.outcome_validator import validate_recommendations_against_outcomes


class TestDecisionQualityKPIs(unittest.TestCase):
    def setUp(self):
        # TP, FP, FN, TN = 2, 1, 1, 2  (total 6)
        self.y_true = [
            "RETEST_BENEFICIAL",
            "RETEST_BENEFICIAL",
            "PERSISTENT_FAILURE",
            "RETEST_BENEFICIAL",
            "PERSISTENT_FAILURE",
            "PERSISTENT_FAILURE",
        ]
        self.recs = [
            "RETEST",
            "RETEST",
            "RETEST",
            "DON'T RETEST",
            "DON'T RETEST",
            "DON'T RETEST",
        ]

    def test_confusion_and_named_errors(self):
        q = calculate_decision_quality(self.y_true, self.recs)
        self.assertEqual(q["tp"], 2)
        self.assertEqual(q["fp"], 1)
        self.assertEqual(q["fn"], 1)
        self.assertEqual(q["tn"], 2)
        self.assertEqual(q["correct_retests"], 2)
        self.assertEqual(q["unnecessary_retests"], 1)
        self.assertEqual(q["missed_opportunities"], 1)
        self.assertEqual(q["correct_skips"], 2)

    def test_accuracy_precision_recall_specificity(self):
        q = calculate_decision_quality(self.y_true, self.recs)
        self.assertAlmostEqual(q["accuracy"], 4 / 6)
        self.assertAlmostEqual(q["precision"], 2 / 3)
        self.assertAlmostEqual(q["recall"], 2 / 3)
        self.assertAlmostEqual(q["specificity"], 2 / 3)
        self.assertAlmostEqual(q["unnecessary_retests_pct"], 1 / 6 * 100)
        self.assertAlmostEqual(q["missed_opportunities_pct"], 1 / 6 * 100)

    def test_rejects_raw_probabilities(self):
        with self.assertRaises(ValueError):
            calculate_decision_quality(self.y_true, [0.8, 0.2, 0.4, 0.1, 0.9, 0.05])

    def test_outcome_validator_event_split(self):
        events = pd.DataFrame({
            "Device_ID": [f"D{i}" for i in range(6)],
            "Failure_Event": list(range(6)),
        })
        result = validate_recommendations_against_outcomes(self.y_true, self.recs, events=events)
        self.assertEqual(len(result["correct_retest_events"]), 2)
        self.assertEqual(len(result["unnecessary_retest_events"]), 1)
        self.assertEqual(len(result["missed_opportunity_events"]), 1)
        self.assertEqual(len(result["correct_skip_events"]), 2)


if __name__ == "__main__":
    unittest.main()
