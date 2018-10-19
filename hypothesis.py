import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind_from_stats
import matplotlib.pyplot as plt

# Read Files
consoleSales = pd.read_csv('consoleSales.csv')
videoSales = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')

# Drop the NA values
videoSales = videoSales.dropna()

# Select all cases where PUBLISHER are nintende, microsoft, sony
videoSales = videoSales.loc[videoSales['Publisher'].isin(['Nintendo','Microsoft Game Studios'
                            ,'Sony Computer Entertainment'])]

# Select all cases where yeaR is greater than 2007
videoSales = videoSales [videoSales['Year_of_Release'] > 2007.0]
videoSales.drop(['Rating', 'Other_Sales', 'JP_Sales', 'EU_Sales', 'NA_Sales','Genre'], axis=1, inplace=True)

# Group by year and publisher and sum of Global Sales
videoSalesSum = videoSales.groupby(['Year_of_Release','Publisher'])['Global_Sales'].sum().reset_index()
bar1=sns.barplot(x=videoSalesSum['Year_of_Release'],
            y=videoSalesSum['Global_Sales'],
            hue=videoSalesSum['Publisher'], data=videoSalesSum,
            hue_order=['Nintendo','Sony Computer Entertainment','Microsoft Game Studios'])
plt.show(bar1)

# Console sales bar chart
bar2=sns.barplot(x=consoleSales['Year'],
            y=consoleSales['ConsoleUnitSold'],
            hue=consoleSales['Company'])
plt.show(bar2)

# Group by two year and puslisher mean and sum
videoSalesMean = videoSales.groupby(['Year_of_Release','Publisher']).mean().reset_index()

# Merge the two file based on year and publisher
merged = pd.concat([videoSalesMean, consoleSales], axis=1)
merged.drop(['Company','Year'] ,axis=1, inplace=True)
print(merged.isnull().sum())

# Correlation heat map
corr = merged.corr()
corrMap = sns.heatmap(corr, annot=True, cbar=True, cmap="RdYlGn")
plt.show(corrMap)

# Merge the global sales and console unit sales
videoSalesSum = videoSalesSum.rename(columns={'Year_of_Release': 'Year','Publisher': 'Company' })
console_and_Global_Sale = pd.merge(consoleSales,videoSalesSum,
                                   how='inner',
                                   on=['Year','Company'])
console_and_Global_Sale.drop(['Year','Company'], axis=1, inplace=True)

# Merge the global sales and console unit sales
videoSalesSum = videoSalesSum.rename(columns={'Year_of_Release':'Year', 'Publisher': 'Company' })

console_and_Global_Sale = pd.merge(consoleSales,videoSalesSum,
                                   how='inner',
                                   on=['Year','Company'])
console_and_Global_Sale.drop(['Year','Company'], axis=1, inplace=True)

# CHI SQUARE TEST OF INDEPENDENCE
chi2, p, dof, expected = stats.chi2_contingency(console_and_Global_Sale)
print(expected)
print(p)
