---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.13
  nbformat: 4
  nbformat_minor: 1
---

::: {.cell .markdown id="GtYqxlPx4ISt"}
# Libraries
:::

::: {.cell .code execution_count="1" id="e1hv9xIh3ILy"}
``` python
# Libraries
## sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


## basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
```
:::

::: {.cell .markdown id="v2E_QNXN6b6g"}
# Fundamentals
:::

::: {.cell .markdown id="AbBO6TIa4uBI"}
## Import Data

I am interested in learning patterns of stocks by analyzing and
clustering fundamental indicators of stocks data in December 2016.
According to dataset source in Kaggle:

`fundamentals.csv`: metrics extracted from annual SEC 10K fillings
(2012-2016), should be enough to derive most of popular fundamental
indicators.

You can access all data from this
[link](https://www.kaggle.com/code/uknowabhishek/nyse-fundamentals-analysis-and-k-means-clustering/notebook).
:::

::: {.cell .code execution_count="2" id="wafyYtii4fNB"}
``` python
# read the data
df_fundamentals = pd.read_csv('fundamentals.csv')
```
:::

::: {.cell .code execution_count="3" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":357}" id="1NTHcyUy6fwp" outputId="9caff554-6c5e-4e9e-d93b-fcd2ae6b98ff"}
``` python
# first few rows of the data
df_fundamentals.head()
```

::: {.output .execute_result execution_count="3"}
``` json
{"type":"dataframe","variable_name":"df_fundamentals"}
```
:::
:::

::: {.cell .code execution_count="4" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="QcnPSx6W7J47" outputId="9814be3c-6812-43bc-f91d-08bb9d50624f"}
``` python
# number of columns and rows in the fundamentals data
length = df_fundamentals.shape

print(f"There are {length[0]} rows and {length[1]} columns in the fundamental indicators dataset")
```

::: {.output .stream .stdout}
    There are 1781 rows and 79 columns in the fundamental indicators dataset
:::
:::

::: {.cell .code execution_count="5" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ifjy2onE7ufS" outputId="b4b8af25-c104-4972-f3de-4e698198dd95"}
``` python
# information about data and its datatypes
df_fundamentals.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1781 entries, 0 to 1780
    Data columns (total 79 columns):
     #   Column                                               Non-Null Count  Dtype  
    ---  ------                                               --------------  -----  
     0   Unnamed: 0                                           1781 non-null   int64  
     1   Ticker Symbol                                        1781 non-null   object 
     2   Period Ending                                        1781 non-null   object 
     3   Accounts Payable                                     1781 non-null   float64
     4   Accounts Receivable                                  1781 non-null   float64
     5   Add'l income/expense items                           1781 non-null   float64
     6   After Tax ROE                                        1781 non-null   float64
     7   Capital Expenditures                                 1781 non-null   float64
     8   Capital Surplus                                      1781 non-null   float64
     9   Cash Ratio                                           1482 non-null   float64
     10  Cash and Cash Equivalents                            1781 non-null   float64
     11  Changes in Inventories                               1781 non-null   float64
     12  Common Stocks                                        1781 non-null   float64
     13  Cost of Revenue                                      1781 non-null   float64
     14  Current Ratio                                        1482 non-null   float64
     15  Deferred Asset Charges                               1781 non-null   float64
     16  Deferred Liability Charges                           1781 non-null   float64
     17  Depreciation                                         1781 non-null   float64
     18  Earnings Before Interest and Tax                     1781 non-null   float64
     19  Earnings Before Tax                                  1781 non-null   float64
     20  Effect of Exchange Rate                              1781 non-null   float64
     21  Equity Earnings/Loss Unconsolidated Subsidiary       1781 non-null   float64
     22  Fixed Assets                                         1781 non-null   float64
     23  Goodwill                                             1781 non-null   float64
     24  Gross Margin                                         1781 non-null   float64
     25  Gross Profit                                         1781 non-null   float64
     26  Income Tax                                           1781 non-null   float64
     27  Intangible Assets                                    1781 non-null   float64
     28  Interest Expense                                     1781 non-null   float64
     29  Inventory                                            1781 non-null   float64
     30  Investments                                          1781 non-null   float64
     31  Liabilities                                          1781 non-null   float64
     32  Long-Term Debt                                       1781 non-null   float64
     33  Long-Term Investments                                1781 non-null   float64
     34  Minority Interest                                    1781 non-null   float64
     35  Misc. Stocks                                         1781 non-null   float64
     36  Net Borrowings                                       1781 non-null   float64
     37  Net Cash Flow                                        1781 non-null   float64
     38  Net Cash Flow-Operating                              1781 non-null   float64
     39  Net Cash Flows-Financing                             1781 non-null   float64
     40  Net Cash Flows-Investing                             1781 non-null   float64
     41  Net Income                                           1781 non-null   float64
     42  Net Income Adjustments                               1781 non-null   float64
     43  Net Income Applicable to Common Shareholders         1781 non-null   float64
     44  Net Income-Cont. Operations                          1781 non-null   float64
     45  Net Receivables                                      1781 non-null   float64
     46  Non-Recurring Items                                  1781 non-null   float64
     47  Operating Income                                     1781 non-null   float64
     48  Operating Margin                                     1781 non-null   float64
     49  Other Assets                                         1781 non-null   float64
     50  Other Current Assets                                 1781 non-null   float64
     51  Other Current Liabilities                            1781 non-null   float64
     52  Other Equity                                         1781 non-null   float64
     53  Other Financing Activities                           1781 non-null   float64
     54  Other Investing Activities                           1781 non-null   float64
     55  Other Liabilities                                    1781 non-null   float64
     56  Other Operating Activities                           1781 non-null   float64
     57  Other Operating Items                                1781 non-null   float64
     58  Pre-Tax Margin                                       1781 non-null   float64
     59  Pre-Tax ROE                                          1781 non-null   float64
     60  Profit Margin                                        1781 non-null   float64
     61  Quick Ratio                                          1482 non-null   float64
     62  Research and Development                             1781 non-null   float64
     63  Retained Earnings                                    1781 non-null   float64
     64  Sale and Purchase of Stock                           1781 non-null   float64
     65  Sales, General and Admin.                            1781 non-null   float64
     66  Short-Term Debt / Current Portion of Long-Term Debt  1781 non-null   float64
     67  Short-Term Investments                               1781 non-null   float64
     68  Total Assets                                         1781 non-null   float64
     69  Total Current Assets                                 1781 non-null   float64
     70  Total Current Liabilities                            1781 non-null   float64
     71  Total Equity                                         1781 non-null   float64
     72  Total Liabilities                                    1781 non-null   float64
     73  Total Liabilities & Equity                           1781 non-null   float64
     74  Total Revenue                                        1781 non-null   float64
     75  Treasury Stock                                       1781 non-null   float64
     76  For Year                                             1608 non-null   float64
     77  Earnings Per Share                                   1562 non-null   float64
     78  Estimated Shares Outstanding                         1562 non-null   float64
    dtypes: float64(76), int64(1), object(2)
    memory usage: 1.1+ MB
:::
:::

::: {.cell .markdown id="wu5GNVoe736D"}
There are 1781 rows and 79 columns in the dataset. I need to apply
`Principal Component Analysis` (PCA) to reduce dimensionality of the
data. Furthermore, there are `Uknown` and `Period Ending` columns which
are going to be irrelevant for clustering, so I have to drop them. But
before I drop `Period Ending` column, I need to use this column to
filter data for only December 2016 fundamental metrics.
:::

::: {.cell .markdown id="CN7Xswzz8VPN"}
## Data Cleaning
:::

::: {.cell .code execution_count="7" id="QnMKQnmnhw9o"}
``` python
# convert `Period Ending` column to datetime
df_fundamentals['Period Ending'] = pd.to_datetime(df_fundamentals['Period Ending'])

# filter fundamental metrics data to include stock indicators of December 2016
filtered_fundamentals = df_fundamentals[(df_fundamentals['Period Ending'].dt.year == 2016) & (df_fundamentals['Period Ending'].dt.month == 12)]
```
:::

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":357}" id="CcTg7IKhjVrB" outputId="993dde7b-6f53-4756-b842-61d109a6da18"}
``` python
# check
filtered_fundamentals.head()
```

::: {.output .execute_result execution_count="8"}
``` json
{"type":"dataframe","variable_name":"filtered_fundamentals"}
```
:::
:::

::: {.cell .code execution_count="9" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="e2Ox-Vlrjq7t" outputId="c08a703e-95f0-4669-9b2e-4ca33c7fa42f"}
``` python
# check #2
filtered_fundamentals['Period Ending'].value_counts()
```

::: {.output .execute_result execution_count="9"}
    2016-12-31    97
    2016-12-02     1
    2016-12-30     1
    Name: Period Ending, dtype: int64
:::
:::

::: {.cell .code execution_count="10" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="IQ6n1K0Dl-51" outputId="a106facb-16aa-4b3b-dbd5-142bab27601d"}
``` python
# check 3
filtered_fundamentals.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99 entries, 27 to 1780
    Data columns (total 79 columns):
     #   Column                                               Non-Null Count  Dtype         
    ---  ------                                               --------------  -----         
     0   Unnamed: 0                                           99 non-null     int64         
     1   Ticker Symbol                                        99 non-null     object        
     2   Period Ending                                        99 non-null     datetime64[ns]
     3   Accounts Payable                                     99 non-null     float64       
     4   Accounts Receivable                                  99 non-null     float64       
     5   Add'l income/expense items                           99 non-null     float64       
     6   After Tax ROE                                        99 non-null     float64       
     7   Capital Expenditures                                 99 non-null     float64       
     8   Capital Surplus                                      99 non-null     float64       
     9   Cash Ratio                                           86 non-null     float64       
     10  Cash and Cash Equivalents                            99 non-null     float64       
     11  Changes in Inventories                               99 non-null     float64       
     12  Common Stocks                                        99 non-null     float64       
     13  Cost of Revenue                                      99 non-null     float64       
     14  Current Ratio                                        86 non-null     float64       
     15  Deferred Asset Charges                               99 non-null     float64       
     16  Deferred Liability Charges                           99 non-null     float64       
     17  Depreciation                                         99 non-null     float64       
     18  Earnings Before Interest and Tax                     99 non-null     float64       
     19  Earnings Before Tax                                  99 non-null     float64       
     20  Effect of Exchange Rate                              99 non-null     float64       
     21  Equity Earnings/Loss Unconsolidated Subsidiary       99 non-null     float64       
     22  Fixed Assets                                         99 non-null     float64       
     23  Goodwill                                             99 non-null     float64       
     24  Gross Margin                                         99 non-null     float64       
     25  Gross Profit                                         99 non-null     float64       
     26  Income Tax                                           99 non-null     float64       
     27  Intangible Assets                                    99 non-null     float64       
     28  Interest Expense                                     99 non-null     float64       
     29  Inventory                                            99 non-null     float64       
     30  Investments                                          99 non-null     float64       
     31  Liabilities                                          99 non-null     float64       
     32  Long-Term Debt                                       99 non-null     float64       
     33  Long-Term Investments                                99 non-null     float64       
     34  Minority Interest                                    99 non-null     float64       
     35  Misc. Stocks                                         99 non-null     float64       
     36  Net Borrowings                                       99 non-null     float64       
     37  Net Cash Flow                                        99 non-null     float64       
     38  Net Cash Flow-Operating                              99 non-null     float64       
     39  Net Cash Flows-Financing                             99 non-null     float64       
     40  Net Cash Flows-Investing                             99 non-null     float64       
     41  Net Income                                           99 non-null     float64       
     42  Net Income Adjustments                               99 non-null     float64       
     43  Net Income Applicable to Common Shareholders         99 non-null     float64       
     44  Net Income-Cont. Operations                          99 non-null     float64       
     45  Net Receivables                                      99 non-null     float64       
     46  Non-Recurring Items                                  99 non-null     float64       
     47  Operating Income                                     99 non-null     float64       
     48  Operating Margin                                     99 non-null     float64       
     49  Other Assets                                         99 non-null     float64       
     50  Other Current Assets                                 99 non-null     float64       
     51  Other Current Liabilities                            99 non-null     float64       
     52  Other Equity                                         99 non-null     float64       
     53  Other Financing Activities                           99 non-null     float64       
     54  Other Investing Activities                           99 non-null     float64       
     55  Other Liabilities                                    99 non-null     float64       
     56  Other Operating Activities                           99 non-null     float64       
     57  Other Operating Items                                99 non-null     float64       
     58  Pre-Tax Margin                                       99 non-null     float64       
     59  Pre-Tax ROE                                          99 non-null     float64       
     60  Profit Margin                                        99 non-null     float64       
     61  Quick Ratio                                          86 non-null     float64       
     62  Research and Development                             99 non-null     float64       
     63  Retained Earnings                                    99 non-null     float64       
     64  Sale and Purchase of Stock                           99 non-null     float64       
     65  Sales, General and Admin.                            99 non-null     float64       
     66  Short-Term Debt / Current Portion of Long-Term Debt  99 non-null     float64       
     67  Short-Term Investments                               99 non-null     float64       
     68  Total Assets                                         99 non-null     float64       
     69  Total Current Assets                                 99 non-null     float64       
     70  Total Current Liabilities                            99 non-null     float64       
     71  Total Equity                                         99 non-null     float64       
     72  Total Liabilities                                    99 non-null     float64       
     73  Total Liabilities & Equity                           99 non-null     float64       
     74  Total Revenue                                        99 non-null     float64       
     75  Treasury Stock                                       99 non-null     float64       
     76  For Year                                             0 non-null      float64       
     77  Earnings Per Share                                   0 non-null      float64       
     78  Estimated Shares Outstanding                         0 non-null      float64       
    dtypes: datetime64[ns](1), float64(76), int64(1), object(1)
    memory usage: 61.9+ KB
:::
:::

::: {.cell .code execution_count="11" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="eO2yemL-hAs9" outputId="64d40b51-c51d-4125-c408-dad389c7b2a2"}
``` python
# check 4
# we have 99 different stocks
filtered_fundamentals['Ticker Symbol'].value_counts()
```

::: {.output .execute_result execution_count="11"}
    ADBE     1
    PG       1
    PEP      1
    PCG      1
    OMC      1
            ..
    DOV      1
    DLPH     1
    DISCK    1
    DISCA    1
    ZTS      1
    Name: Ticker Symbol, Length: 99, dtype: int64
:::
:::

::: {.cell .code execution_count="12" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Om6yb7-373Oe" outputId="cfd11d6c-313a-4a81-a30a-b2a298dfe7e6"}
``` python
# drop uneccesary columns
## Unnamed column
filtered_fundamentals.drop(columns = ['Unnamed: 0', 'Period Ending'], axis = 1, inplace = True)
```

::: {.output .stream .stderr}
    <ipython-input-12-4f0692c2dfef>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_fundamentals.drop(columns = ['Unnamed: 0', 'Period Ending'], axis = 1, inplace = True)
:::
:::

::: {.cell .code execution_count="13" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ZzqweDLZ8mhZ" outputId="f7f3ff62-90be-4931-801c-0b8465730e64"}
``` python
# check
# first few rows of the data
filtered_fundamentals.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 99 entries, 27 to 1780
    Data columns (total 77 columns):
     #   Column                                               Non-Null Count  Dtype  
    ---  ------                                               --------------  -----  
     0   Ticker Symbol                                        99 non-null     object 
     1   Accounts Payable                                     99 non-null     float64
     2   Accounts Receivable                                  99 non-null     float64
     3   Add'l income/expense items                           99 non-null     float64
     4   After Tax ROE                                        99 non-null     float64
     5   Capital Expenditures                                 99 non-null     float64
     6   Capital Surplus                                      99 non-null     float64
     7   Cash Ratio                                           86 non-null     float64
     8   Cash and Cash Equivalents                            99 non-null     float64
     9   Changes in Inventories                               99 non-null     float64
     10  Common Stocks                                        99 non-null     float64
     11  Cost of Revenue                                      99 non-null     float64
     12  Current Ratio                                        86 non-null     float64
     13  Deferred Asset Charges                               99 non-null     float64
     14  Deferred Liability Charges                           99 non-null     float64
     15  Depreciation                                         99 non-null     float64
     16  Earnings Before Interest and Tax                     99 non-null     float64
     17  Earnings Before Tax                                  99 non-null     float64
     18  Effect of Exchange Rate                              99 non-null     float64
     19  Equity Earnings/Loss Unconsolidated Subsidiary       99 non-null     float64
     20  Fixed Assets                                         99 non-null     float64
     21  Goodwill                                             99 non-null     float64
     22  Gross Margin                                         99 non-null     float64
     23  Gross Profit                                         99 non-null     float64
     24  Income Tax                                           99 non-null     float64
     25  Intangible Assets                                    99 non-null     float64
     26  Interest Expense                                     99 non-null     float64
     27  Inventory                                            99 non-null     float64
     28  Investments                                          99 non-null     float64
     29  Liabilities                                          99 non-null     float64
     30  Long-Term Debt                                       99 non-null     float64
     31  Long-Term Investments                                99 non-null     float64
     32  Minority Interest                                    99 non-null     float64
     33  Misc. Stocks                                         99 non-null     float64
     34  Net Borrowings                                       99 non-null     float64
     35  Net Cash Flow                                        99 non-null     float64
     36  Net Cash Flow-Operating                              99 non-null     float64
     37  Net Cash Flows-Financing                             99 non-null     float64
     38  Net Cash Flows-Investing                             99 non-null     float64
     39  Net Income                                           99 non-null     float64
     40  Net Income Adjustments                               99 non-null     float64
     41  Net Income Applicable to Common Shareholders         99 non-null     float64
     42  Net Income-Cont. Operations                          99 non-null     float64
     43  Net Receivables                                      99 non-null     float64
     44  Non-Recurring Items                                  99 non-null     float64
     45  Operating Income                                     99 non-null     float64
     46  Operating Margin                                     99 non-null     float64
     47  Other Assets                                         99 non-null     float64
     48  Other Current Assets                                 99 non-null     float64
     49  Other Current Liabilities                            99 non-null     float64
     50  Other Equity                                         99 non-null     float64
     51  Other Financing Activities                           99 non-null     float64
     52  Other Investing Activities                           99 non-null     float64
     53  Other Liabilities                                    99 non-null     float64
     54  Other Operating Activities                           99 non-null     float64
     55  Other Operating Items                                99 non-null     float64
     56  Pre-Tax Margin                                       99 non-null     float64
     57  Pre-Tax ROE                                          99 non-null     float64
     58  Profit Margin                                        99 non-null     float64
     59  Quick Ratio                                          86 non-null     float64
     60  Research and Development                             99 non-null     float64
     61  Retained Earnings                                    99 non-null     float64
     62  Sale and Purchase of Stock                           99 non-null     float64
     63  Sales, General and Admin.                            99 non-null     float64
     64  Short-Term Debt / Current Portion of Long-Term Debt  99 non-null     float64
     65  Short-Term Investments                               99 non-null     float64
     66  Total Assets                                         99 non-null     float64
     67  Total Current Assets                                 99 non-null     float64
     68  Total Current Liabilities                            99 non-null     float64
     69  Total Equity                                         99 non-null     float64
     70  Total Liabilities                                    99 non-null     float64
     71  Total Liabilities & Equity                           99 non-null     float64
     72  Total Revenue                                        99 non-null     float64
     73  Treasury Stock                                       99 non-null     float64
     74  For Year                                             0 non-null      float64
     75  Earnings Per Share                                   0 non-null      float64
     76  Estimated Shares Outstanding                         0 non-null      float64
    dtypes: float64(76), object(1)
    memory usage: 60.3+ KB
:::
:::

::: {.cell .markdown id="csqc6JcC9c8J"}
There is also `Ticker Symbol` column that I need to manage in order to
properly apply PCA as PCA should be used on numerical values.

Another important step is to remove null values in `CurrentRation`
column, I am going to replace all null values with 0.
:::

::: {.cell .code execution_count="14" id="HuribHIB90NV"}
``` python
# make ticker symbols column as an index
filtered_fundamentals = filtered_fundamentals.set_index(['Ticker Symbol'])
# transform everything to numeric datatype
filtered_fundamentals = filtered_fundamentals.apply(pd.to_numeric, errors = 'coerce')
# fill null values with 0 in dataframe
filtered_fundamentals = filtered_fundamentals.fillna(0)
```
:::

::: {.cell .code execution_count="15" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Rgn6MJb990Kv" outputId="e4b542a9-7f82-4da1-a5b5-42ffc4f66caa"}
``` python
# check number of na rows
filtered_fundamentals.isnull().sum()
# there are no null values anymore
```

::: {.output .execute_result execution_count="15"}
    Accounts Payable                0
    Accounts Receivable             0
    Add'l income/expense items      0
    After Tax ROE                   0
    Capital Expenditures            0
                                   ..
    Total Revenue                   0
    Treasury Stock                  0
    For Year                        0
    Earnings Per Share              0
    Estimated Shares Outstanding    0
    Length: 76, dtype: int64
:::
:::

::: {.cell .code execution_count="16" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="bXtm5E5a9zjI" outputId="fd78abf5-e0ab-4a25-d93e-f61ee836f828"}
``` python
# check datatypes
filtered_fundamentals.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 99 entries, ADBE to ZTS
    Data columns (total 76 columns):
     #   Column                                               Non-Null Count  Dtype  
    ---  ------                                               --------------  -----  
     0   Accounts Payable                                     99 non-null     float64
     1   Accounts Receivable                                  99 non-null     float64
     2   Add'l income/expense items                           99 non-null     float64
     3   After Tax ROE                                        99 non-null     float64
     4   Capital Expenditures                                 99 non-null     float64
     5   Capital Surplus                                      99 non-null     float64
     6   Cash Ratio                                           99 non-null     float64
     7   Cash and Cash Equivalents                            99 non-null     float64
     8   Changes in Inventories                               99 non-null     float64
     9   Common Stocks                                        99 non-null     float64
     10  Cost of Revenue                                      99 non-null     float64
     11  Current Ratio                                        99 non-null     float64
     12  Deferred Asset Charges                               99 non-null     float64
     13  Deferred Liability Charges                           99 non-null     float64
     14  Depreciation                                         99 non-null     float64
     15  Earnings Before Interest and Tax                     99 non-null     float64
     16  Earnings Before Tax                                  99 non-null     float64
     17  Effect of Exchange Rate                              99 non-null     float64
     18  Equity Earnings/Loss Unconsolidated Subsidiary       99 non-null     float64
     19  Fixed Assets                                         99 non-null     float64
     20  Goodwill                                             99 non-null     float64
     21  Gross Margin                                         99 non-null     float64
     22  Gross Profit                                         99 non-null     float64
     23  Income Tax                                           99 non-null     float64
     24  Intangible Assets                                    99 non-null     float64
     25  Interest Expense                                     99 non-null     float64
     26  Inventory                                            99 non-null     float64
     27  Investments                                          99 non-null     float64
     28  Liabilities                                          99 non-null     float64
     29  Long-Term Debt                                       99 non-null     float64
     30  Long-Term Investments                                99 non-null     float64
     31  Minority Interest                                    99 non-null     float64
     32  Misc. Stocks                                         99 non-null     float64
     33  Net Borrowings                                       99 non-null     float64
     34  Net Cash Flow                                        99 non-null     float64
     35  Net Cash Flow-Operating                              99 non-null     float64
     36  Net Cash Flows-Financing                             99 non-null     float64
     37  Net Cash Flows-Investing                             99 non-null     float64
     38  Net Income                                           99 non-null     float64
     39  Net Income Adjustments                               99 non-null     float64
     40  Net Income Applicable to Common Shareholders         99 non-null     float64
     41  Net Income-Cont. Operations                          99 non-null     float64
     42  Net Receivables                                      99 non-null     float64
     43  Non-Recurring Items                                  99 non-null     float64
     44  Operating Income                                     99 non-null     float64
     45  Operating Margin                                     99 non-null     float64
     46  Other Assets                                         99 non-null     float64
     47  Other Current Assets                                 99 non-null     float64
     48  Other Current Liabilities                            99 non-null     float64
     49  Other Equity                                         99 non-null     float64
     50  Other Financing Activities                           99 non-null     float64
     51  Other Investing Activities                           99 non-null     float64
     52  Other Liabilities                                    99 non-null     float64
     53  Other Operating Activities                           99 non-null     float64
     54  Other Operating Items                                99 non-null     float64
     55  Pre-Tax Margin                                       99 non-null     float64
     56  Pre-Tax ROE                                          99 non-null     float64
     57  Profit Margin                                        99 non-null     float64
     58  Quick Ratio                                          99 non-null     float64
     59  Research and Development                             99 non-null     float64
     60  Retained Earnings                                    99 non-null     float64
     61  Sale and Purchase of Stock                           99 non-null     float64
     62  Sales, General and Admin.                            99 non-null     float64
     63  Short-Term Debt / Current Portion of Long-Term Debt  99 non-null     float64
     64  Short-Term Investments                               99 non-null     float64
     65  Total Assets                                         99 non-null     float64
     66  Total Current Assets                                 99 non-null     float64
     67  Total Current Liabilities                            99 non-null     float64
     68  Total Equity                                         99 non-null     float64
     69  Total Liabilities                                    99 non-null     float64
     70  Total Liabilities & Equity                           99 non-null     float64
     71  Total Revenue                                        99 non-null     float64
     72  Treasury Stock                                       99 non-null     float64
     73  For Year                                             99 non-null     float64
     74  Earnings Per Share                                   99 non-null     float64
     75  Estimated Shares Outstanding                         99 non-null     float64
    dtypes: float64(76)
    memory usage: 59.6+ KB
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":339}" id="MYYlZmIv-59c" outputId="93cb59db-ebde-4f4d-f3ae-bb1d7004a952"}
``` python
# first few rows of the dataset
filtered_fundamentals.head()
```

::: {.output .execute_result execution_count="15"}
``` json
{"type":"dataframe","variable_name":"filtered_fundamentals"}
```
:::
:::

::: {.cell .markdown id="OVGO4YDp-4EG"}
There are no null values in the data and all datatypes are numeric now.
There are now 99 rows and 76 columns. I need to apply PCA and reduce
dimensionality while capturing most of the variance in the data.
:::

::: {.cell .markdown id="hYZuabEYmddD"}
## Principal Component Analysis

Before I can apply PCA on my data, all numeric variables should be
normalized for a proper analysis. That\'s why I am going to use a
standard normalization, or z-statistic normalization for all my numeric
variables.
:::

::: {.cell .code execution_count="17" id="YDfEr4J8maMZ"}
``` python
# standard scaler
scaler = StandardScaler()

# object that will have standardized variables
fundamentals_norm = pd.DataFrame(scaler.fit_transform(filtered_fundamentals), columns = filtered_fundamentals.columns)
```
:::

::: {.cell .code execution_count="18" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":384}" id="mJ_k-6xonarh" outputId="0603ecd3-5894-44ef-c5e6-880e50f91649"}
``` python
# summary to show if normalization worked
fundamentals_norm.describe()
```

::: {.output .execute_result execution_count="18"}
``` json
{"type":"dataframe"}
```
:::
:::

::: {.cell .code execution_count="19" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":270}" id="2XP46U14nmCU" outputId="37ae5846-daf7-4611-9451-08cc523f6372"}
``` python
# head() to show few rows of the normalized data
fundamentals_norm.head()
```

::: {.output .execute_result execution_count="19"}
``` json
{"type":"dataframe","variable_name":"fundamentals_norm"}
```
:::
:::

::: {.cell .markdown id="H2su1u4moIsX"}
As all of the variables in the data normalized, I need to choose number
of components (variables) that I will include in the data. In order to
do it, I will compare number of components vs Cumulative Explained
Variance and I will choose optimal number that has the most of data
variance saved.
:::

::: {.cell .code execution_count="20" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="d1pQLMhmoZVa" outputId="7f59db1f-0f8c-4945-9610-dc7315271766"}
``` python
# model with a specific number of components for PCA
## n_components = number of components
model_pca = PCA(n_components=25)

# fit the dataa to the pca model
model_pca.fit_transform(fundamentals_norm)

cumulative_variance = model_pca.explained_variance_ratio_.cumsum()
highest_cumulative_variance = cumulative_variance[-1]
n_components = len(cumulative_variance)


# plot
plt.figure(figsize = (12, 10))
plt.plot(range(1, n_components + 1), cumulative_variance)

## parameters
### title
plt.title('Number of components VS Cumulative Explained Variance', fontweight = 'bold', fontsize = 18)

### xlabels
plt.xlabel('Number of components', fontsize = 16)
plt.xticks([1, 5, 10, 15, 20, 25, 30], fontsize = 16)

### ylabels
plt.ylabel('Cumulative explained variance', fontsize = 16 )
plt.yticks(fontsize = 16)


### vertical lines from grids to determine number of components with the highest explained variation
plt.grid(True,
         which = 'major',
         linestyle = '--',
        axis = 'x')

### annotation to find higest variance possible and corresponding number of components
plt.annotate(f"Highest cumulative variance possible: {highest_cumulative_variance: .2f}",
             xy = (n_components, highest_cumulative_variance),
             xytext = (n_components - 5, highest_cumulative_variance - 0.1),
             arrowprops = dict(facecolor = 'red'),
             horizontalalignment = 'center')
plt.show()


### dataframe for more clear understanding
all_components = np.arange(1, n_components + 1)

pd.DataFrame({
    'Number of Components' : all_components,
    'Cumulative Variance for Component': cumulative_variance
})
```

::: {.output .display_data}
![](vertopal_1e43fb906b67491c962e6dd45f6cc5d9/842b26b2172dbbdf2cb137e34eacf611ea9a95aa.png)
:::

::: {.output .execute_result execution_count="20"}
``` json
{"summary":"{\n  \"name\": \"})\",\n  \"rows\": 25,\n  \"fields\": [\n    {\n      \"column\": \"Number of Components\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 7,\n        \"min\": 1,\n        \"max\": 25,\n        \"num_unique_values\": 25,\n        \"samples\": [\n          9,\n          17,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Cumulative Variance for Component\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.19955604621497347,\n        \"min\": 0.2864954467842965,\n        \"max\": 0.9684391723349536,\n        \"num_unique_values\": 25,\n        \"samples\": [\n          0.7368306506149693,\n          0.911105667249178,\n          0.2864954467842965\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {.cell .markdown id="yFqXJeH9x24D"}
At most, I can save \~97% of data variance using 25 components. However,
if I take 20 components, I will save \~94% of data variance and I will
reduce the dimensionality of the data even more. While it is lower than
97%, the difference is only 3%, so I am going to choose `20` components
for the sake of dimensionality reduction.
:::

::: {.cell .markdown id="7OcNhN3l0RVJ"}
## Determining Number of K clusters for K-means clustering

As I identified optimal number of components, I am going to use this
reduced data to find an optiminal number of clusters and then produce a
K-means clustering.
:::

::: {.cell .code execution_count="35" id="ZjFt5MbJynAY"}
``` python
# fit main data to pca with 20 components
pca = PCA(n_components = 20)
data = pca.fit_transform(fundamentals_norm)
```
:::

::: {.cell .code execution_count="36" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="qtoJVDIb0qoH" outputId="32db15b8-f198-4f6f-ca46-a48ceab26842"}
``` python
## sse - SUM OF SQUARED ERRORS
## record sss for each number of clusters
sse = []
for k in range(1, 25):
  print(k)
  kmeans =  KMeans(n_clusters = k, random_state = 1042)
  kmeans.fit(data)
  sse.append(kmeans.inertia_)
```

::: {.output .stream .stdout}
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::

::: {.output .stream .stdout}
    11
    12
    13
    14
    15
    16
    17
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::

::: {.output .stream .stdout}
    18
    19
    20
    21
    22
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::

::: {.output .stream .stdout}
    23
    24
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::
:::

::: {.cell .code execution_count="37" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="vY9C2L5_1azi" outputId="a91748e5-8e31-41a5-fec0-76f1b1114c65"}
``` python
## check
sse
```

::: {.output .execute_result execution_count="37"}
    [6787.94099290186,
     5279.219600522962,
     4586.072804190557,
     4293.172657166553,
     3922.607401598773,
     3616.070641312063,
     3345.4432309063695,
     3067.981932287598,
     2698.2077478249107,
     2551.3905147619234,
     2330.2671342980566,
     2093.2643650819327,
     1873.871349310599,
     1648.658007879532,
     1544.6117407433285,
     1392.4925942770806,
     1293.7602400437502,
     1113.3925323655951,
     1063.862753932816,
     941.3566162391026,
     864.9282364200395,
     819.4087444022276,
     777.214711159224,
     688.6582637762617]
:::
:::

::: {.cell .code execution_count="38" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":889}" id="l7bTCs5I1ejm" outputId="4e8fe99d-f0cb-4935-ade9-a7031429f95b"}
``` python
# plot SSE vs number of clusters
plt.figure(figsize = (12, 10))
plt.plot(range(1, 25), sse, marker='o')

## parameters
### title
plt.title('SSE vs Number of Clusters', fontweight = 'bold', fontsize = 18)

### xlabels
plt.xlabel("Number of Clusters", fontsize = 16)
plt.xticks(range(1, 25), fontsize = 16)

### ylabels
plt.ylabel("SSE", fontsize = 16)
plt.yticks(fontsize = 16)

plt.show()
```

::: {.output .display_data}
![](vertopal_1e43fb906b67491c962e6dd45f6cc5d9/d1ce40be6d3b21cb3ab2b2a1ee287c15ffc09c5d.png)
:::
:::

::: {.cell .markdown id="fK7yStmD2tMp"}
Above visualization shows that SSE highly decreases till \~3 clusters,
then it marginally decreases, as clusters increase. Marginall decrease
is not something I am looking for as I am intending to cluster
datapoints into distinct groups. That\'s why I will choose 3 clusters.
:::

::: {.cell .markdown id="YfZGJfJP83eh"}
## Clustering
:::

::: {.cell .code execution_count="39" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="dhKFF6ns2r7t" outputId="b1eb0c34-2ce3-4865-9750-23e379e168ef"}
``` python
# clustering model
model_clusters = KMeans(n_clusters = 3, random_state = 1042)

# cluster labels with reduced data
cluster_labels = model_clusters.fit_predict(data)


# cluster classes for observations
print(cluster_labels[0:10])
```

::: {.output .stream .stdout}
    [1 1 1 1 1 0 0 1 1 0]
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::
:::

::: {.cell .code execution_count="63" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":716}" id="RQTQTA5m2xrO" outputId="0ace7e3e-e104-4782-97da-eae37c32ce67"}
``` python
# initial plot without centroids
ig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.scatter(data[:,0], data[:,1], c=cluster_labels)
ax.set_title("K-Means Clustering Results with K=3")
```

::: {.output .execute_result execution_count="63"}
    Text(0.5, 1.0, 'K-Means Clustering Results with K=3')
:::

::: {.output .display_data}
![](vertopal_1e43fb906b67491c962e6dd45f6cc5d9/ed2bfa71b4b764250e8adf2aa5016693189fbc44.png)
:::
:::

::: {.cell .markdown id="usFOCG5Ki9ao"}
With 20 components and 3 clusters, it is clear that some yellow and
purple groups are overlapping. I want to try and to cluster groups in a
better and more distinct clusters.

I assume that such overlapping might be related to noise and to a number
of components, as I only saved 93% of explained data variance. I will
use 25 components with 3 clusters with 97% of explained data variance to
see if it is going to improve the clustering process.
:::

::: {.cell .code execution_count="41" id="bnHnnyD3jjSP"}
``` python
# fit main data to pca with 20 components
pca = PCA(n_components = 25)
data = pca.fit_transform(fundamentals_norm)
```
:::

::: {.cell .code execution_count="43" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="-Lmi53PdkYRp" outputId="9adbae27-22a4-4780-9c42-a1b9eedfe04a"}
``` python
# clustering model
model_clusters = KMeans(n_clusters = 3, random_state = 1042)

# cluster labels with reduced data
cluster_labels = model_clusters.fit_predict(data)


# cluster classes for observations
print(cluster_labels[0:10])
```

::: {.output .stream .stdout}
    [2 2 2 2 2 0 0 2 2 0]
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::
:::

::: {.cell .code execution_count="64" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":716}" id="OeddLC6blJWz" outputId="ab6782c1-e783-4b5f-cc2c-fc78246be5b4"}
``` python
# initial plot without centroids
ig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.scatter(data[:,0], data[:,1], c=cluster_labels)
ax.set_title("K-Means Clustering Results with K=3")
```

::: {.output .execute_result execution_count="64"}
    Text(0.5, 1.0, 'K-Means Clustering Results with K=3')
:::

::: {.output .display_data}
![](vertopal_1e43fb906b67491c962e6dd45f6cc5d9/ed2bfa71b4b764250e8adf2aa5016693189fbc44.png)
:::
:::

::: {.cell .markdown id="5G7fmYBimd6i"}
K-Means Clustering with 25 components and 3 clusters is a little bit
better than K-Means Clustering with 20 components. There is 1
observation which changes its cluster color in second graph, which leads
to more clear and not overlapping groups. While it is an insignificant
improvement, I am going to leave 25 components for this data clustering.

The last part is to produce a finalized graph with centroids in the
center of each group.
:::

::: {.cell .markdown id="Vznuf1Y69PPI"}
### Generate Clusters for each cluster group
:::

::: {.cell .code execution_count="46" id="4-W--TR7CxHs"}
``` python
# data with our clusters
data

# define clusters in the data
centroids = model_clusters.cluster_centers_
```
:::

::: {.cell .code execution_count="65" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":858}" id="CgngXPkD8O2n" outputId="cef017a0-239d-4999-b61f-45d8a78ae298"}
``` python
# data with each cluster as unique value
each_cluster = np.unique(cluster_labels)

# figure size
plt.figure(figsize = (12, 10))

for cluster in each_cluster:
    # Filter by each cluster
    cluster_data = data[cluster_labels == cluster]

    # Plot the data points
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster}", alpha=0.5)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=250, c='black', label='Centroids')

# parameters
## title
plt.title('K-Means Clustering Results with K=3 and centroids', fontweight = 'bold', fontsize = 18)

## legend
plt.legend(fontsize = 16)

## grid
plt.grid(True)

# show the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_1e43fb906b67491c962e6dd45f6cc5d9/564cc2a363d11dffde8b46ff66713fe232d472d4.png)
:::
:::

::: {.cell .code id="BJ9VTmP-EMkw"}
``` python
```
:::
