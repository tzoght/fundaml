import pandas as pd

def print_basic_eda(df,shape=True,info=True,describe=True,null=True,na=True,duplicate=True, dtypes=False, corr= False, usage=False):
    """ Print basic EDA of a dataframe.
    Parameters
    ----------
    
    Returns
    -------
    
    Examples
    --------
    >>> from mltz_base import eda
    >>> eda.print_basic_eda(df)
    """
    sep = "----------------------------------------\n"
    if shape:
        print(f"{sep} Shape: {df.shape}")
    if info:
        print(sep)
        df.info()
    if null:
        print(f"{sep} Null:\n {df.isnull().sum()}")
    if describe:
        print(f"{sep} Describe:\n {df.describe(include='all')}")
    if usage:
        print(f"{sep} Usage:\n {df.memory_usage(deep=True)}")
    if na:
        print(f"{sep} NA:\n {df.isna().sum()}")
    if duplicate:
        print(f"{sep} Duplicate:\n {df.duplicated().sum()}")
        print(f"{sep} Duplicated:\n {df.duplicated()}")
    if corr:
        print(f"{sep} Correlation:\n {df.corr()}")
    if dtypes:
        print(f"{sep} Dtypes:\n {df.dtypes}")