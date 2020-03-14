# Overiew
<p>The <b>dfsummary</b> package is a simple EDA helper that enables you to quickly execute those typical, initial reviews of your data (including descriptive statistics), all within a single method.</p>

<p>This package also enables users to quickly produce formatted, production-level histograms, correlation heatmaps and boxplots (with a swarmplot option) of numeric data fields without needing to invest the time required to format these plots.

# Getting Started

## Installation
The dfsummary package is available on PyPi:

```pip install dfsummary```

## Requirements
This package requires the following installations:
- python: 3.7.5
- pandas: 1.0.1
- numpy: 1.18.1
- matplotlib: 3.1.2
- seaborn: 0.10.0

## Usage
Below is a simple example that uses the dfsummary package on scikit-learn's boston dataset.  Note, additional method parameters are available and explained in the method docstrings.


```
# Import the package
from dfsummary import DfSummary

# Load the boston dataset into a dataframe
data = datasets.load_boston()
df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
df['target'] = data['target']

# Instantiate data frame summary (DfSummary object)
dfs=DfSummary(df)

# Return initial EDA and descriptive statistics
dfs.return_summary()

# Return formatted histograms of all numeric data columns
dfs.return_histograms()

# Return a formatted correlation heatmap of all numeric data columns
dfs.return_heatmap()

# Return formatted boxplots of all numeric data columns
dfs.return_boxplots()
```
