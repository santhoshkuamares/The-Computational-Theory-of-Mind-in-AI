"""Test the CPU scoring and statistical helpers on small known examples.

Run from the repository root. The checks compare weighted calculations with
explicit row expansion and test paired swaps, Holm correction, parsing and
invalid-answer handling; they do not execute a model.
"""

import unittest, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from analyse import *


class AnalysisTests(unittest.TestCase):

    def test_fixed_class_f1_and_bootstrap_against_row_expansion(self):
        """Compare weighted cluster calculations with explicitly repeated question
        rows and scikit-learn F1. Agreement checks the efficient bootstrap
        implementation against a separately calculated result, including
        invalid predictions.
        """
        rows = []
        for c, gt, pr in [
            ("a", "A", "A"),
            ("a", "B", "A"),
            ("b", "A", "B"),
            ("c", "B", "B"),
            ("c", "B", "INVALID"),
        ]:
            rows.append(
                {
                    "cluster": c,
                    "family": "desire",
                    "reference": gt,
                    **{s: pr for s in SYSTEMS},
                }
            )
        df = pd.DataFrame(rows)
        groups, families, n, c, classes = cluster_statistics(df)
        weights = np.array([[2, 0, 1], [0, 3, 0], [1, 1, 1]], float)
        result = metrics_for_weights(weights, n, c, classes, families)
        for j, w in enumerate(weights):
            expanded = pd.concat(
                [df[df.cluster == g] for g, k in zip(groups, w) for _ in range(int(k))]
            )
            self.assertAlmostEqual(
                result["accuracy"][0, j],
                (expanded.reference == expanded.base_direct).mean(),
            )
            self.assertAlmostEqual(
                result["mean_family_f1"][0, j],
                f1_score(
                    expanded.reference,
                    expanded.base_direct,
                    labels=["A", "B"],
                    average="macro",
                    zero_division=0,
                ),
            )

    def test_signflip_exact_enumeration(self):
        """Enumerate every possible sign assignment for a small paired example and
        compare its p-value with signflip_p. Including a zero contribution
        checks that dropping such clusters leaves the result unchanged.
        """
        d = np.array([0.2, -0.1, 0.3, 0])
        expected = np.mean(
            [
                abs(np.dot(d, s)) >= abs(d.sum()) - 1e-12
                for s in itertools.product([-1, 1], repeat=4)
            ]
        )
        self.assertEqual(signflip_p(d)[0], expected)

    def test_zero_difference(self):
        """Require a p-value of one when every paired cluster contribution is
        zero. Identical accuracy contributions provide no evidence of a
        difference under this test.
        """
        self.assertEqual(signflip_p([0, 0, 0])[0], 1)

    def test_holm(self):
        """Compare Holm-adjusted p-values with a small example whose answers are
        known. This checks both the multiplicity correction and restoration of
        the original order.
        """
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_duplicates_rejected(self):
        """Pass two records with the same ID and require the indexing helper to
        reject them. Duplicate IDs would otherwise overwrite observations and
        corrupt paired comparisons.
        """
        with self.assertRaises(AssertionError):
            index([{"record_id": "x"}, {"record_id": "x"}])

    def test_corrected_location_keys_and_ambiguous_output(self):
        """Check that location scoring uses explicit original and moved metadata
        and retains an ambiguous response as an error label. The example guards
        the correction that restored complete OpenToM scoring.
        """
        ref = {
            "family": "location_fg_fo",
            "family_detail": "location_fg_fo",
            "answer": "the pantry",
            "original_place": "the pantry",
            "move_to_place": "a drawer",
        }
        self.assertEqual(parse_open("drawer", ref), ("1", "2"))
        self.assertEqual(parse_open("unknown", ref), ("1", "3"))

    def test_absent_class_zero_and_invalid_not_dropped(self):
        """Score an invalid prediction in a dataset with an absent target class.
        Require zero accuracy and fixed-class macro F1 rather than dropping the
        invalid answer or changing the class set.
        """
        df = pd.DataFrame(
            [
                {
                    "cluster": "a",
                    "family": "desire",
                    "reference": "A",
                    **{s: "INVALID" for s in SYSTEMS},
                }
            ]
        )
        g, f, n, c, cl = cluster_statistics(df)
        m = metrics_for_weights(np.ones((1, 1)), n, c, cl, f)
        self.assertEqual(m["accuracy"][0, 0], 0)
        self.assertEqual(m["mean_family_f1"][0, 0], 0)

    def test_json_wrapped(self):
        """Check that the JSON reader can recover an object surrounded by ordinary
        text. This is needed when saved generated responses contain prose
        around their structured record.
        """
        self.assertEqual(json_object('text {"a": 1} end'), {"a": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
