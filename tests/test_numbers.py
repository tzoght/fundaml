from mltz_base.numbers import binary 
from mltz_base.numbers import calc_spacing

def test_binary():
    assert binary(1.25) == None
    
def test_calc_spacing():
    assert calc_spacing(25.0) == 3.552713678800501e-15