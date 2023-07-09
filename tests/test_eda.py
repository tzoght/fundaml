from fundaml.eda import print_basic_eda
import pandas as pd

def test_basic_eda():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert print_basic_eda(df) == None