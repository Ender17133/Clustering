# Libraries


```python
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

# Fundamentals

## Import Data
I am interested in learning patterns of stocks by analyzing and clustering fundamental indicators of stocks data in December 2016. According to dataset source in Kaggle:

`fundamentals.csv`: metrics extracted from annual SEC 10K fillings (2012-2016), should be enough to derive most of popular fundamental indicators.

You can access all data from this [link](https://www.kaggle.com/code/uknowabhishek/nyse-fundamentals-analysis-and-k-means-clustering/notebook).


```python
# read the data
df_fundamentals = pd.read_csv('fundamentals.csv')
```


```python
# first few rows of the data
df_fundamentals.head()
```





  <div id="df-0ec65789-2b95-4532-9ac7-d254b3c72995" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Ticker Symbol</th>
      <th>Period Ending</th>
      <th>Accounts Payable</th>
      <th>Accounts Receivable</th>
      <th>Add'l income/expense items</th>
      <th>After Tax ROE</th>
      <th>Capital Expenditures</th>
      <th>Capital Surplus</th>
      <th>Cash Ratio</th>
      <th>...</th>
      <th>Total Current Assets</th>
      <th>Total Current Liabilities</th>
      <th>Total Equity</th>
      <th>Total Liabilities</th>
      <th>Total Liabilities &amp; Equity</th>
      <th>Total Revenue</th>
      <th>Treasury Stock</th>
      <th>For Year</th>
      <th>Earnings Per Share</th>
      <th>Estimated Shares Outstanding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>AAL</td>
      <td>2012-12-31</td>
      <td>3.068000e+09</td>
      <td>-222000000.0</td>
      <td>-1.961000e+09</td>
      <td>23.0</td>
      <td>-1.888000e+09</td>
      <td>4.695000e+09</td>
      <td>53.0</td>
      <td>...</td>
      <td>7.072000e+09</td>
      <td>9.011000e+09</td>
      <td>-7.987000e+09</td>
      <td>2.489100e+10</td>
      <td>1.690400e+10</td>
      <td>2.485500e+10</td>
      <td>-367000000.0</td>
      <td>2012.0</td>
      <td>-5.60</td>
      <td>3.350000e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>AAL</td>
      <td>2013-12-31</td>
      <td>4.975000e+09</td>
      <td>-93000000.0</td>
      <td>-2.723000e+09</td>
      <td>67.0</td>
      <td>-3.114000e+09</td>
      <td>1.059200e+10</td>
      <td>75.0</td>
      <td>...</td>
      <td>1.432300e+10</td>
      <td>1.380600e+10</td>
      <td>-2.731000e+09</td>
      <td>4.500900e+10</td>
      <td>4.227800e+10</td>
      <td>2.674300e+10</td>
      <td>0.0</td>
      <td>2013.0</td>
      <td>-11.25</td>
      <td>1.630222e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>AAL</td>
      <td>2014-12-31</td>
      <td>4.668000e+09</td>
      <td>-160000000.0</td>
      <td>-1.500000e+08</td>
      <td>143.0</td>
      <td>-5.311000e+09</td>
      <td>1.513500e+10</td>
      <td>60.0</td>
      <td>...</td>
      <td>1.175000e+10</td>
      <td>1.340400e+10</td>
      <td>2.021000e+09</td>
      <td>4.120400e+10</td>
      <td>4.322500e+10</td>
      <td>4.265000e+10</td>
      <td>0.0</td>
      <td>2014.0</td>
      <td>4.02</td>
      <td>7.169154e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>AAL</td>
      <td>2015-12-31</td>
      <td>5.102000e+09</td>
      <td>352000000.0</td>
      <td>-7.080000e+08</td>
      <td>135.0</td>
      <td>-6.151000e+09</td>
      <td>1.159100e+10</td>
      <td>51.0</td>
      <td>...</td>
      <td>9.985000e+09</td>
      <td>1.360500e+10</td>
      <td>5.635000e+09</td>
      <td>4.278000e+10</td>
      <td>4.841500e+10</td>
      <td>4.099000e+10</td>
      <td>0.0</td>
      <td>2015.0</td>
      <td>11.39</td>
      <td>6.681299e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>AAP</td>
      <td>2012-12-29</td>
      <td>2.409453e+09</td>
      <td>-89482000.0</td>
      <td>6.000000e+05</td>
      <td>32.0</td>
      <td>-2.711820e+08</td>
      <td>5.202150e+08</td>
      <td>23.0</td>
      <td>...</td>
      <td>3.184200e+09</td>
      <td>2.559638e+09</td>
      <td>1.210694e+09</td>
      <td>3.403120e+09</td>
      <td>4.613814e+09</td>
      <td>6.205003e+09</td>
      <td>-27095000.0</td>
      <td>2012.0</td>
      <td>5.29</td>
      <td>7.328355e+07</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 79 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0ec65789-2b95-4532-9ac7-d254b3c72995')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0ec65789-2b95-4532-9ac7-d254b3c72995 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0ec65789-2b95-4532-9ac7-d254b3c72995');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1921b9c5-e542-4e5e-9b04-36d4eea1cb9d">
  <button class="colab-df-quickchart" onclick="quickchart('df-1921b9c5-e542-4e5e-9b04-36d4eea1cb9d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1921b9c5-e542-4e5e-9b04-36d4eea1cb9d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
# number of columns and rows in the fundamentals data
length = df_fundamentals.shape

print(f"There are {length[0]} rows and {length[1]} columns in the fundamental indicators dataset")
```

    There are 1781 rows and 79 columns in the fundamental indicators dataset
    


```python
# information about data and its datatypes
df_fundamentals.info()
```

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
    

There are 1781 rows and 79 columns in the dataset. I need to apply `Principal Component Analysis` (PCA) to reduce dimensionality of the data. Furthermore, there are `Uknown` and `Period Ending` columns which are going to be irrelevant for clustering, so I have to drop them. But before I drop `Period Ending` column, I need to use this column to filter data for only December 2016 fundamental metrics.

## Data Cleaning


```python
# convert `Period Ending` column to datetime
df_fundamentals['Period Ending'] = pd.to_datetime(df_fundamentals['Period Ending'])

# filter fundamental metrics data to include stock indicators of December 2016
filtered_fundamentals = df_fundamentals[(df_fundamentals['Period Ending'].dt.year == 2016) & (df_fundamentals['Period Ending'].dt.month == 12)]
```


```python
# check
filtered_fundamentals.head()
```





  <div id="df-18e41ff1-91b5-4428-ae1b-6669b26f9f63" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Ticker Symbol</th>
      <th>Period Ending</th>
      <th>Accounts Payable</th>
      <th>Accounts Receivable</th>
      <th>Add'l income/expense items</th>
      <th>After Tax ROE</th>
      <th>Capital Expenditures</th>
      <th>Capital Surplus</th>
      <th>Cash Ratio</th>
      <th>...</th>
      <th>Total Current Assets</th>
      <th>Total Current Liabilities</th>
      <th>Total Equity</th>
      <th>Total Liabilities</th>
      <th>Total Liabilities &amp; Equity</th>
      <th>Total Revenue</th>
      <th>Treasury Stock</th>
      <th>For Year</th>
      <th>Earnings Per Share</th>
      <th>Estimated Shares Outstanding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>ADBE</td>
      <td>2016-12-02</td>
      <td>8.660160e+08</td>
      <td>-160416000.0</td>
      <td>11978000.0</td>
      <td>16.0</td>
      <td>-203805000.0</td>
      <td>4.616331e+09</td>
      <td>169.0</td>
      <td>...</td>
      <td>5.839774e+09</td>
      <td>2.811635e+09</td>
      <td>7.424835e+09</td>
      <td>5.282279e+09</td>
      <td>1.270711e+10</td>
      <td>5.854430e+09</td>
      <td>-5.132472e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>67</th>
      <td>67</td>
      <td>AIZ</td>
      <td>2016-12-31</td>
      <td>2.080962e+09</td>
      <td>9275000.0</td>
      <td>-23031000.0</td>
      <td>14.0</td>
      <td>-85233000.0</td>
      <td>3.175867e+09</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.098100e+09</td>
      <td>2.561103e+10</td>
      <td>2.970913e+10</td>
      <td>7.531780e+09</td>
      <td>-4.470551e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>71</th>
      <td>71</td>
      <td>AJG</td>
      <td>2016-12-31</td>
      <td>3.768200e+09</td>
      <td>-240000000.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>-217800000.0</td>
      <td>3.265500e+09</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.596600e+09</td>
      <td>7.893000e+09</td>
      <td>1.148960e+10</td>
      <td>5.594800e+09</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>91</th>
      <td>91</td>
      <td>ALLE</td>
      <td>2016-12-31</td>
      <td>3.814000e+08</td>
      <td>-19800000.0</td>
      <td>-66200000.0</td>
      <td>202.0</td>
      <td>-42500000.0</td>
      <td>0.000000e+00</td>
      <td>73.0</td>
      <td>...</td>
      <td>8.293000e+08</td>
      <td>4.296000e+08</td>
      <td>1.133000e+08</td>
      <td>2.134100e+09</td>
      <td>2.247400e+09</td>
      <td>2.238000e+09</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>95</th>
      <td>95</td>
      <td>ALXN</td>
      <td>2016-12-31</td>
      <td>5.720000e+08</td>
      <td>-122000000.0</td>
      <td>6000000.0</td>
      <td>5.0</td>
      <td>-333000000.0</td>
      <td>7.957000e+09</td>
      <td>157.0</td>
      <td>...</td>
      <td>2.578000e+09</td>
      <td>8.230000e+08</td>
      <td>8.694000e+09</td>
      <td>4.559000e+09</td>
      <td>1.325300e+10</td>
      <td>3.084000e+09</td>
      <td>-1.141000e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 79 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-18e41ff1-91b5-4428-ae1b-6669b26f9f63')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-18e41ff1-91b5-4428-ae1b-6669b26f9f63 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-18e41ff1-91b5-4428-ae1b-6669b26f9f63');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e3a7ae60-2704-4f98-85cc-19752bc1ee85">
  <button class="colab-df-quickchart" onclick="quickchart('df-e3a7ae60-2704-4f98-85cc-19752bc1ee85')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e3a7ae60-2704-4f98-85cc-19752bc1ee85 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
# check #2
filtered_fundamentals['Period Ending'].value_counts()
```




    2016-12-31    97
    2016-12-02     1
    2016-12-30     1
    Name: Period Ending, dtype: int64




```python
# check 3
filtered_fundamentals.info()
```

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
    


```python
# check 4
# we have 99 different stocks
filtered_fundamentals['Ticker Symbol'].value_counts()
```




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




```python
# drop uneccesary columns
## Unnamed column
filtered_fundamentals.drop(columns = ['Unnamed: 0', 'Period Ending'], axis = 1, inplace = True)
```

    <ipython-input-12-4f0692c2dfef>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_fundamentals.drop(columns = ['Unnamed: 0', 'Period Ending'], axis = 1, inplace = True)
    


```python
# check
# first few rows of the data
filtered_fundamentals.info()
```

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
    

There is also `Ticker Symbol` column that I need to manage in order to properly apply PCA as PCA should be used on numerical values.

Another important step is to remove null values in `CurrentRation` column, I am going to replace all null values with 0.


```python
# make ticker symbols column as an index
filtered_fundamentals = filtered_fundamentals.set_index(['Ticker Symbol'])
# transform everything to numeric datatype
filtered_fundamentals = filtered_fundamentals.apply(pd.to_numeric, errors = 'coerce')
# fill null values with 0 in dataframe
filtered_fundamentals = filtered_fundamentals.fillna(0)
```


```python
# check number of na rows
filtered_fundamentals.isnull().sum()
# there are no null values anymore
```




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




```python
# check datatypes
filtered_fundamentals.info()
```

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
    


```python
# first few rows of the dataset
filtered_fundamentals.head()
```





  <div id="df-975cad77-ccc6-4001-9962-61d3316c1615" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accounts Payable</th>
      <th>Accounts Receivable</th>
      <th>Add'l income/expense items</th>
      <th>After Tax ROE</th>
      <th>Capital Expenditures</th>
      <th>Capital Surplus</th>
      <th>Cash Ratio</th>
      <th>Cash and Cash Equivalents</th>
      <th>Changes in Inventories</th>
      <th>Common Stocks</th>
      <th>...</th>
      <th>Total Current Assets</th>
      <th>Total Current Liabilities</th>
      <th>Total Equity</th>
      <th>Total Liabilities</th>
      <th>Total Liabilities &amp; Equity</th>
      <th>Total Revenue</th>
      <th>Treasury Stock</th>
      <th>For Year</th>
      <th>Earnings Per Share</th>
      <th>Estimated Shares Outstanding</th>
    </tr>
    <tr>
      <th>Ticker Symbol</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADBE</th>
      <td>8.660160e+08</td>
      <td>-160416000.0</td>
      <td>11978000.0</td>
      <td>16.0</td>
      <td>-203805000.0</td>
      <td>4.616331e+09</td>
      <td>169.0</td>
      <td>1.011315e+09</td>
      <td>0.0</td>
      <td>61000.0</td>
      <td>...</td>
      <td>5.839774e+09</td>
      <td>2.811635e+09</td>
      <td>7.424835e+09</td>
      <td>5.282279e+09</td>
      <td>1.270711e+10</td>
      <td>5.854430e+09</td>
      <td>-5.132472e+09</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>AIZ</th>
      <td>2.080962e+09</td>
      <td>9275000.0</td>
      <td>-23031000.0</td>
      <td>14.0</td>
      <td>-85233000.0</td>
      <td>3.175867e+09</td>
      <td>0.0</td>
      <td>1.031971e+09</td>
      <td>4579000.0</td>
      <td>1504000.0</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.098100e+09</td>
      <td>2.561103e+10</td>
      <td>2.970913e+10</td>
      <td>7.531780e+09</td>
      <td>-4.470551e+09</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>AJG</th>
      <td>3.768200e+09</td>
      <td>-240000000.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>-217800000.0</td>
      <td>3.265500e+09</td>
      <td>0.0</td>
      <td>1.937600e+09</td>
      <td>0.0</td>
      <td>178300000.0</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.596600e+09</td>
      <td>7.893000e+09</td>
      <td>1.148960e+10</td>
      <td>5.594800e+09</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ALLE</th>
      <td>3.814000e+08</td>
      <td>-19800000.0</td>
      <td>-66200000.0</td>
      <td>202.0</td>
      <td>-42500000.0</td>
      <td>0.000000e+00</td>
      <td>73.0</td>
      <td>3.124000e+08</td>
      <td>-15600000.0</td>
      <td>1000000.0</td>
      <td>...</td>
      <td>8.293000e+08</td>
      <td>4.296000e+08</td>
      <td>1.133000e+08</td>
      <td>2.134100e+09</td>
      <td>2.247400e+09</td>
      <td>2.238000e+09</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ALXN</th>
      <td>5.720000e+08</td>
      <td>-122000000.0</td>
      <td>6000000.0</td>
      <td>5.0</td>
      <td>-333000000.0</td>
      <td>7.957000e+09</td>
      <td>157.0</td>
      <td>9.660000e+08</td>
      <td>-84000000.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.578000e+09</td>
      <td>8.230000e+08</td>
      <td>8.694000e+09</td>
      <td>4.559000e+09</td>
      <td>1.325300e+10</td>
      <td>3.084000e+09</td>
      <td>-1.141000e+09</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 76 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-975cad77-ccc6-4001-9962-61d3316c1615')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-975cad77-ccc6-4001-9962-61d3316c1615 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-975cad77-ccc6-4001-9962-61d3316c1615');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-bf90fc17-5eb9-49c9-95cb-47a6e312b1a8">
  <button class="colab-df-quickchart" onclick="quickchart('df-bf90fc17-5eb9-49c9-95cb-47a6e312b1a8')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-bf90fc17-5eb9-49c9-95cb-47a6e312b1a8 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




There are no null values in the data and all datatypes are numeric now. There are now 99 rows and 76 columns. I need to apply PCA and reduce dimensionality while capturing most of the variance in the data.

## Principal Component Analysis
Before I can apply PCA on my data, all numeric variables should be normalized for a proper analysis. That's why I am going to use a standard normalization, or z-statistic normalization for all my numeric variables.


```python
# standard scaler
scaler = StandardScaler()

# object that will have standardized variables
fundamentals_norm = pd.DataFrame(scaler.fit_transform(filtered_fundamentals), columns = filtered_fundamentals.columns)
```


```python
# summary to show if normalization worked
fundamentals_norm.describe()
```





  <div id="df-df39658d-fd53-461d-bc73-1db4da51b29b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accounts Payable</th>
      <th>Accounts Receivable</th>
      <th>Add'l income/expense items</th>
      <th>After Tax ROE</th>
      <th>Capital Expenditures</th>
      <th>Capital Surplus</th>
      <th>Cash Ratio</th>
      <th>Cash and Cash Equivalents</th>
      <th>Changes in Inventories</th>
      <th>Common Stocks</th>
      <th>...</th>
      <th>Total Current Assets</th>
      <th>Total Current Liabilities</th>
      <th>Total Equity</th>
      <th>Total Liabilities</th>
      <th>Total Liabilities &amp; Equity</th>
      <th>Total Revenue</th>
      <th>Treasury Stock</th>
      <th>For Year</th>
      <th>Earnings Per Share</th>
      <th>Estimated Shares Outstanding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>...</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>9.900000e+01</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.812887e-17</td>
      <td>-1.233581e-17</td>
      <td>-1.345725e-17</td>
      <td>-1.082888e-17</td>
      <td>6.840768e-17</td>
      <td>2.242875e-18</td>
      <td>1.489409e-18</td>
      <td>-7.008984e-18</td>
      <td>2.355019e-17</td>
      <td>3.476456e-17</td>
      <td>...</td>
      <td>8.971499e-18</td>
      <td>4.990396e-17</td>
      <td>-3.700743e-17</td>
      <td>-1.570012e-17</td>
      <td>9.083643e-17</td>
      <td>3.308240e-17</td>
      <td>2.467162e-17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>...</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>1.005089e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.572983e-01</td>
      <td>-5.842321e+00</td>
      <td>-2.389050e+00</td>
      <td>-2.619738e-01</td>
      <td>-4.123848e+00</td>
      <td>-6.530631e-01</td>
      <td>-5.538382e-01</td>
      <td>-2.505287e-01</td>
      <td>-3.093597e+00</td>
      <td>-3.110193e-01</td>
      <td>...</td>
      <td>-5.846669e-01</td>
      <td>-5.083298e-01</td>
      <td>-1.874118e+00</td>
      <td>-6.560521e-01</td>
      <td>-7.478530e-01</td>
      <td>-6.081754e-01</td>
      <td>-3.459925e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-4.633142e-01</td>
      <td>-1.400747e-02</td>
      <td>-2.121788e-01</td>
      <td>-2.125131e-01</td>
      <td>-1.538561e-01</td>
      <td>-6.043762e-01</td>
      <td>-4.250123e-01</td>
      <td>-2.232486e-01</td>
      <td>-9.878170e-02</td>
      <td>-3.106388e-01</td>
      <td>...</td>
      <td>-4.594576e-01</td>
      <td>-4.326361e-01</td>
      <td>-6.051258e-01</td>
      <td>-5.233013e-01</td>
      <td>-5.808073e-01</td>
      <td>-4.759785e-01</td>
      <td>-1.142553e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-3.337615e-01</td>
      <td>2.268168e-01</td>
      <td>-7.192636e-02</td>
      <td>-1.855345e-01</td>
      <td>3.911792e-01</td>
      <td>-3.347610e-01</td>
      <td>-2.878750e-01</td>
      <td>-1.867831e-01</td>
      <td>-1.734517e-02</td>
      <td>-3.077844e-01</td>
      <td>...</td>
      <td>-3.788483e-01</td>
      <td>-3.148813e-01</td>
      <td>-3.589456e-01</td>
      <td>-3.920960e-01</td>
      <td>-3.715790e-01</td>
      <td>-3.388296e-01</td>
      <td>4.699595e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-1.315102e-01</td>
      <td>3.238790e-01</td>
      <td>5.320550e-02</td>
      <td>-1.293291e-01</td>
      <td>5.880183e-01</td>
      <td>1.498103e-01</td>
      <td>7.366879e-02</td>
      <td>-6.959544e-02</td>
      <td>-1.734517e-02</td>
      <td>-2.313821e-01</td>
      <td>...</td>
      <td>-4.058344e-02</td>
      <td>-7.565692e-02</td>
      <td>1.496974e-01</td>
      <td>1.100645e-01</td>
      <td>1.484804e-01</td>
      <td>-6.665996e-02</td>
      <td>5.857355e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.510885e+00</td>
      <td>1.970275e+00</td>
      <td>6.380328e+00</td>
      <td>9.283947e+00</td>
      <td>6.862839e-01</td>
      <td>4.369852e+00</td>
      <td>7.956985e+00</td>
      <td>9.662578e+00</td>
      <td>8.083163e+00</td>
      <td>5.546935e+00</td>
      <td>...</td>
      <td>6.000373e+00</td>
      <td>5.819472e+00</td>
      <td>4.549106e+00</td>
      <td>4.466455e+00</td>
      <td>4.060852e+00</td>
      <td>4.432801e+00</td>
      <td>5.857355e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 76 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-df39658d-fd53-461d-bc73-1db4da51b29b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-df39658d-fd53-461d-bc73-1db4da51b29b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-df39658d-fd53-461d-bc73-1db4da51b29b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-03848b5b-2c61-4bc2-b9c6-cf38801c6df7">
  <button class="colab-df-quickchart" onclick="quickchart('df-03848b5b-2c61-4bc2-b9c6-cf38801c6df7')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-03848b5b-2c61-4bc2-b9c6-cf38801c6df7 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
# head() to show few rows of the normalized data
fundamentals_norm.head()
```





  <div id="df-b3a2ed4f-c4a5-424d-a98e-c889eb633370" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accounts Payable</th>
      <th>Accounts Receivable</th>
      <th>Add'l income/expense items</th>
      <th>After Tax ROE</th>
      <th>Capital Expenditures</th>
      <th>Capital Surplus</th>
      <th>Cash Ratio</th>
      <th>Cash and Cash Equivalents</th>
      <th>Changes in Inventories</th>
      <th>Common Stocks</th>
      <th>...</th>
      <th>Total Current Assets</th>
      <th>Total Current Liabilities</th>
      <th>Total Equity</th>
      <th>Total Liabilities</th>
      <th>Total Liabilities &amp; Equity</th>
      <th>Total Revenue</th>
      <th>Treasury Stock</th>
      <th>For Year</th>
      <th>Earnings Per Share</th>
      <th>Estimated Shares Outstanding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.465085</td>
      <td>0.030099</td>
      <td>-0.032156</td>
      <td>-0.194527</td>
      <td>0.584432</td>
      <td>-0.097529</td>
      <td>0.850780</td>
      <td>-0.193712</td>
      <td>-0.017345</td>
      <td>-0.311008</td>
      <td>...</td>
      <td>-0.230114</td>
      <td>-0.311262</td>
      <td>-0.237029</td>
      <td>-0.547999</td>
      <td>-0.529710</td>
      <td>-0.469774</td>
      <td>0.010501</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.333761</td>
      <td>0.340865</td>
      <td>-0.158134</td>
      <td>-0.203520</td>
      <td>0.643688</td>
      <td>-0.270876</td>
      <td>-0.553838</td>
      <td>-0.192524</td>
      <td>-0.007467</td>
      <td>-0.310733</td>
      <td>...</td>
      <td>-0.584667</td>
      <td>-0.508330</td>
      <td>-0.507809</td>
      <td>-0.076495</td>
      <td>-0.190354</td>
      <td>-0.423830</td>
      <td>0.084687</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.151388</td>
      <td>-0.115648</td>
      <td>-0.075259</td>
      <td>-0.212513</td>
      <td>0.577438</td>
      <td>-0.260089</td>
      <td>-0.553838</td>
      <td>-0.140442</td>
      <td>-0.017345</td>
      <td>-0.277090</td>
      <td>...</td>
      <td>-0.584667</td>
      <td>-0.508330</td>
      <td>-0.548629</td>
      <td>-0.487446</td>
      <td>-0.554011</td>
      <td>-0.476885</td>
      <td>0.585735</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.517467</td>
      <td>0.287618</td>
      <td>-0.313475</td>
      <td>0.641809</td>
      <td>0.665044</td>
      <td>-0.653063</td>
      <td>0.052890</td>
      <td>-0.233907</td>
      <td>-0.050998</td>
      <td>-0.310829</td>
      <td>...</td>
      <td>-0.534317</td>
      <td>-0.478219</td>
      <td>-0.832153</td>
      <td>-0.621018</td>
      <td>-0.738483</td>
      <td>-0.568831</td>
      <td>0.585735</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.496865</td>
      <td>0.100453</td>
      <td>-0.053668</td>
      <td>-0.243988</td>
      <td>0.519866</td>
      <td>0.304491</td>
      <td>0.751044</td>
      <td>-0.196318</td>
      <td>-0.198555</td>
      <td>-0.311019</td>
      <td>...</td>
      <td>-0.428148</td>
      <td>-0.450646</td>
      <td>-0.133725</td>
      <td>-0.564775</td>
      <td>-0.518814</td>
      <td>-0.545658</td>
      <td>0.457855</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 76 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b3a2ed4f-c4a5-424d-a98e-c889eb633370')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b3a2ed4f-c4a5-424d-a98e-c889eb633370 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b3a2ed4f-c4a5-424d-a98e-c889eb633370');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5d7383e3-13fb-401c-91f4-af61a7ffc942">
  <button class="colab-df-quickchart" onclick="quickchart('df-5d7383e3-13fb-401c-91f4-af61a7ffc942')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5d7383e3-13fb-401c-91f4-af61a7ffc942 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




As all of the variables in the data normalized, I need to choose number of components (variables) that I will include in the data. In order to do it, I will compare number of components vs Cumulative Explained Variance and I will choose optimal number that has the most of data variance saved.


```python
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


    
![png](output_28_0.png)
    






  <div id="df-2af84133-0343-4b73-9d0a-b38cbc50a4f1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number of Components</th>
      <th>Cumulative Variance for Component</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.286495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.379206</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.452330</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.512570</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.570557</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.620420</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.662947</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.702642</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.736831</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.767886</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.797350</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.823335</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.848568</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.866299</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.882846</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>0.898277</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0.911106</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>0.922090</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>0.931740</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>0.939247</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>0.946330</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>0.952661</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>0.958446</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>0.963638</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>0.968439</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2af84133-0343-4b73-9d0a-b38cbc50a4f1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2af84133-0343-4b73-9d0a-b38cbc50a4f1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2af84133-0343-4b73-9d0a-b38cbc50a4f1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f85d7491-54c6-42d4-945e-80c5ef13cd53">
  <button class="colab-df-quickchart" onclick="quickchart('df-f85d7491-54c6-42d4-945e-80c5ef13cd53')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f85d7491-54c6-42d4-945e-80c5ef13cd53 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




At most, I can save ~97% of data variance using 25 components. However, if I take 20 components, I will save ~94% of data variance and I will reduce the dimensionality of the data even more. While it is lower than 97%, the difference is only 3%, so I am going to choose `20` components for the sake of dimensionality reduction.

## Determining Number of K clusters for K-means clustering
As I identified optimal number of components, I am going to use this reduced data to find an optiminal number of clusters and then produce a K-means clustering.


```python
# fit main data to pca with 20 components
pca = PCA(n_components = 20)
data = pca.fit_transform(fundamentals_norm)
```


```python
## sse - SUM OF SQUARED ERRORS
## record sss for each number of clusters
sse = []
for k in range(1, 25):
  print(k)
  kmeans =  KMeans(n_clusters = k, random_state = 1042)
  kmeans.fit(data)
  sse.append(kmeans.inertia_)
```

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
    

    11
    12
    13
    14
    15
    16
    17
    

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
    

    18
    19
    20
    21
    22
    

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
    

    23
    24
    

    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    


```python
## check
sse
```




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




```python
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


    
![png](output_34_0.png)
    


Above visualization shows that SSE highly decreases till ~3 clusters, then it marginally decreases, as clusters increase. Marginall decrease is not something I am looking for as I am intending to cluster datapoints into distinct groups. That's why I will choose 3 clusters.

## Clustering


```python
# clustering model
model_clusters = KMeans(n_clusters = 3, random_state = 1042)

# cluster labels with reduced data
cluster_labels = model_clusters.fit_predict(data)


# cluster classes for observations
print(cluster_labels[0:10])
```

    [1 1 1 1 1 0 0 1 1 0]
    

    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    


```python
# initial plot without centroids
ig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.scatter(data[:,0], data[:,1], c=cluster_labels)
ax.set_title("K-Means Clustering Results with K=3")
```




    Text(0.5, 1.0, 'K-Means Clustering Results with K=3')




    
![png](output_38_1.png)
    


With 20 components and 3 clusters, it is clear that some yellow and purple groups are overlapping. I want to try and to cluster groups in a better and more distinct clusters.

I assume that such overlapping might be related to noise and to a number of components, as I only saved 93% of explained data variance. I will use 25 components with 3 clusters with 97% of explained data variance to see if it is going to improve the clustering process.


```python
# fit main data to pca with 20 components
pca = PCA(n_components = 25)
data = pca.fit_transform(fundamentals_norm)
```


```python
# clustering model
model_clusters = KMeans(n_clusters = 3, random_state = 1042)

# cluster labels with reduced data
cluster_labels = model_clusters.fit_predict(data)


# cluster classes for observations
print(cluster_labels[0:10])
```

    [2 2 2 2 2 0 0 2 2 0]
    

    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    


```python
# initial plot without centroids
ig, ax = plt.subplots(1, 1, figsize=(10,8))
ax.scatter(data[:,0], data[:,1], c=cluster_labels)
ax.set_title("K-Means Clustering Results with K=3")
```




    Text(0.5, 1.0, 'K-Means Clustering Results with K=3')




    
![png](output_42_1.png)
    


K-Means Clustering with 25 components and 3 clusters is a little bit better than K-Means Clustering with 20 components. There is 1 observation which changes its cluster color in second graph, which leads to more clear and not overlapping groups. While it is an insignificant improvement, I am going to leave 25 components for this data clustering.

The last part is to produce a finalized graph with centroids in the center of each group.

### Generate Clusters for each cluster group


```python
# data with our clusters
data

# define clusters in the data
centroids = model_clusters.cluster_centers_

```


```python
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


    
![png](output_46_0.png)
    



```python

```
