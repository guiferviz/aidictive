

import numpy as np

import pandas as pd

from unittest import TestCase

from ..context import recipipe as r


class TestGroupByTransformer(TestCase):

    def test_fit_transform(self):
        # TODO: This is not an unit test...
        df_in = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "blue"],
            "other": [1, 2, 3, 4, 5, 6],
            "amount": [5, 6, 7, 1, 2, 3],
            "index": [3, 4, 5, 0, 1, 2]
        })
        # Set an unordered index to check the correct order of the output.
        df_in.set_index("index", inplace=True)
        t = r.groupby("color", r.scale("amount"))
        df_out = t.fit_transform(df_in)
        norm = 1 / np.std([1, 2, 3])
        expected = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "blue"],
            "other": [1, 2, 3, 4, 5, 6],
            "amount": [-norm, 0, norm, -norm, 0, norm],
            "index": [3, 4, 5, 0, 1, 2]
        })
        expected.set_index("index", inplace=True)
        self.assertTrue(expected.equals(df_out))
