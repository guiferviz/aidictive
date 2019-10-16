
import numpy as np

import pandas as pd

from unittest import TestCase

from ..context import recipipe as r


class TestReplaceTransformer(TestCase):
    def test__transform_columns_text(self):
        df_in = pd.DataFrame({
            "Vowels": ["a", "e", None, "o", "u"]
        })
        t = r.replace(values={None: "i"})
        df_out = t.fit_transform(df_in)
        expected = pd.DataFrame({
            "Vowels": ["a", "e", "i", "o", "u"]
        })
        self.assertTrue(expected.equals(df_out))
