import pandas as pd
import numpy as np

def return_df_summary(df):
    """Calculate descriptive statistics for dataframe columns.

    Args:
    df: Pandas dataframe.  A dataframe containing continuous and/or discrete data.

    Returns:
    A dataframe of descriptive summary statisitcs for df
    """

    df_sum=pd.DataFrame(index=df.columns)
    df_sum['dtype']=df.dtypes
    df_sum['null_count']=df.isnull().sum()
    df_sum['null_percent']=round(df_sum['null_count']/len(df)*100,2)
    df_sum['unique_vals']=df.nunique()
    df_sum['mean']=df.mean()
    df_sum['min']=df.min()
    df_sum['max']=df.max()
    df_sum['std']=df.std()
    df_sum['25%']=df.quantile(0.25)
    df_sum['median']=df.quantile(0.5)
    df_sum['75%']=df.quantile(0.75)
    df_sum['skewness']=df.skew()
    df_sum['kurtosis']=df.kurtosis()

    # Transpose and return df descriptive statistics
    return df_sum.T


def return_heatmap_data(df, method, drop_criteria):
    """Create and return a dictionary of specified correlations between df columns.

    This function is called from the return_heatmap method.
    Pairwise correlations between columns are calculated.

    Args:
    method: string. Method to be used for calculating correlation between df columns
            ('pearson', 'spearman' or 'kendall').
    drop_criteria: string or list of columns in df to be used for determining data to drop.
                  'any_rows': drops any rows in the data containing NaNs
                  'any_cols': drops any columns in the data containing NaNs
                  list: list of columns to be used for subset of drop_na function
                  None: no rows or columns with NaNs are dropped from the data

    Returns:
    heatmap_data: dictionary.  A dictionary of data required for heatmap figure created
    and returned by the return_heatmap method.
    """

    # Initialise dictionary for heatmap data
    heatmap_data={}

    # Based on specified drop_criteria, drop relevant rows or columns and
    # create data subset: corr_data.  Insert number of rows or columns dropped
    # (if any) and text for heatmap plot title in heatmap_data dictionary
    if drop_criteria=='any_rows':
        corr_data=df.dropna(axis=0, how='any')
        heatmap_data['drop_count']=df.shape[0]-corr_data.shape[0]
        heatmap_data['title_text']='rows'
    elif drop_criteria=='any_cols':
        corr_data=df.dropna(axis=1, how='any')
        heatmap_data['drop_count']=df.shape[1]-corr_data.shape[1]
        heatmap_data['title_text']='columns'
    elif type(drop_criteria)==list:
        try:
            corr_data=df.dropna(axis=0, subset=drop_criteria)
            heatmap_data['drop_count']=df.shape[0]-corr_data.shape[0]
            heatmap_data['title_text']= f'rows (subsetted on {", ".join(drop_criteria)}) Null values'
        except (KeyError):
            print('Specified column does not exist in data.  No Null values were dropped.')
            corr_data=df.copy()
            heatmap_data['drop_count']=0
            heatmap_data['title_text']='rows or columns'
    # If invalid or no drop_crtiteria is specified in function call, don't drop any data
    else:
        if drop_criteria != None:
            print('Incorrect drop criteria specified.  No Null values were dropped.')
        corr_data=df.copy()
        heatmap_data['drop_count']=0
        heatmap_data['title_text']='rows or columns'

    # Calculate pairwuse correlations between columns as per specified method
    # and insert into heatmap_data
    heatmap_data['corrs'] = corr_data.corr(method)

    return heatmap_data
