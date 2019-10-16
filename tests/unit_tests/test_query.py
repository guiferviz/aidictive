
import pandas as pd

from unittest import TestCase

from ..context import recipipe as r
from ..fixtures import create_df_all


class TestQueryTransformer(TestCase):

    def test_transform(self):
        df = create_df_all()
        t = r.query("color == 'red'")
        df_out = t.fit_transform(df)
        expected = pd.DataFrame({
            "color": ["red", "red"],
            "price": [1.5, 3.5],
            "amount": [1, 3],
            "index": [0, 2]
        })
        expected.set_index("index", inplace=True)
        self.assertTrue(expected.equals(df_out))
