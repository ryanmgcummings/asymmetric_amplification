# -*- coding: utf-8 -*-
"""
Published: Monday, Nov 13, 2023.

This code produces the analysis in "Asymmetric Amplification and the Consumer Sentiment Gap"
by Ryan Cummings and Neale Mahoney. Note to run the code, you will have to change file locations where necessary
and will need your own API key for FRED. 
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import statsmodels.api as sm
import matplotlib.dates as mdates
import numpy as np

#Load in aggregated consumer sentiment and politcal sentiment.
df = pd.read_excel(r'C:\Users\Ryan Cummings\OneDrive\Documents\Consumer Sentiment\sentiment_october.xlsx', sheet_name="all")
df=df[['date', 'ics_all']]
df.set_index('date', inplace=True)

#Note the six observations before 2006 are just dropped in the data cleaning. 
pol_df = pd.read_excel(r'C:\Users\Ryan Cummings\OneDrive\Documents\Consumer Sentiment\demopoliticalparty202310.xlsx', sheet_name="Partisanship")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Get SP 500, UR, inflation, and PCE into dataframe
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
################################# SP500 ####################################### 

# Download S&P 500 data using yfinance
start_date = "1984-01-01"
end_date = "2023-11-06"
sp500 = yf.download('^GSPC', start=start_date, end=end_date)

# Keep Adjusted Close column and set the date to the first of each month
sp500['date'] = sp500.index.to_period('M').to_timestamp()
sp500_monthly = sp500.groupby('date')['Adj Close'].mean().reset_index()


# Merge the two dataframes
df = pd.merge(df, sp500_monthly, on='date', how='left')

# Calculate the quarterly return of the S&P 500
df['SP500_quarterly_return'] = df['Adj Close'].pct_change(periods=3)

################################# FRED VARS ###################################

#FRED API setup
"""DO NOT FORGET TO DELETE IF POSTING PUBLICLY"""
api_key = '213811f6dd78c7f3d821187a9aad337d'  
fred = Fred(api_key=api_key)

# Pull data from FRED
unemployment_rate = fred.get_series('UNRATE')
cpi_yoy_change = fred.get_series('CPIAUCSL').pct_change(periods=12)  # Year-over-year change
pce = fred.get_series('PCE')
nom_wages=fred.get_series('AHETPI').pct_change(periods=12)
epop=fred.get_series('EMRATIO')

#3 month (annualized) changes for food and gasoline prices
def get_annualized_change(fred, series_id):
    series_data = fred.get_series(series_id)
    return ((1 + series_data.pct_change(periods=3)) ** 4 - 1) * 100

gas_change = get_annualized_change(fred, 'CUUR0000SETB01')
food_change = get_annualized_change(fred, 'CPIUFDSL')

fred_vars=[food_change, gas_change, nom_wages, pce, unemployment_rate]
# Creating a function to change the day part of the date to 1 so merging is fine
def change_day_to_first(date):
    return datetime(date.year, date.month, 1)

for x in fred_vars:
    x.index=x.index.map(change_day_to_first)

# Convert series to dataframes
food_change_df = pd.DataFrame({'date': food_change.index, 'food_change': food_change.values})
gas_change_df = pd.DataFrame({'date': gas_change.index, 'gas_change': gas_change.values})
nom_wages_df = pd.DataFrame({'date': nom_wages.index, 'nom_wages': nom_wages.values})
pce_df = pd.DataFrame({'date': pce.index, 'pce': pce.values})
unemployment_rate_df = pd.DataFrame({'date': unemployment_rate.index, 'unemployment_rate': unemployment_rate.values})
inflation_df = pd.DataFrame({'date': cpi_yoy_change.index, 'inflation': cpi_yoy_change.values})
epop_df = pd.DataFrame({'date': epop.index, 'epop': epop.values})

# Merge each dataframe with the main df
df = pd.merge(df, food_change_df, on='date', how='left')
df = pd.merge(df, gas_change_df, on='date', how='left')
df = pd.merge(df, nom_wages_df, on='date', how='left')
df = pd.merge(df, pce_df, on='date', how='left')
df = pd.merge(df, unemployment_rate_df, on='date', how='left')
df = pd.merge(df, epop_df, on='date', how='left')
df = pd.merge(df, inflation_df, on='date', how='left')

merged_df=df
merged_df.set_index('date', inplace=True)


"""Create Presidential Party Dummy. Note that we assign the Presidential party
in the month after a President of a different party is elected (i.e., in December)
because this is the first month the survey fully incorporates reactions to the 
new Presidental party"""
def assign_party(date):
    if date < pd.Timestamp('2008-12-01'):
        return 'Republican'
    elif pd.Timestamp('2008-12-01') <= date < pd.Timestamp('2016-12-01'):
        return 'Democrat'
    elif pd.Timestamp('2016-12-01') <= date < pd.Timestamp('2020-12-01'):
        return 'Republican'
    else:
        return 'Democrat'
merged_df['date']=merged_df.index   
merged_df['presidential_party'] = merged_df['date'].apply(assign_party)

# Create a dummy variable for presidential party: 1 for Democrats, 0 for Republicans
merged_df['president_party_dummy'] = merged_df['presidential_party'].apply(lambda x: 1 if x == 'Democrat' else 0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Basic sentiment model for all consumers
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Cut off data so it has same time period as political sentiment data
merged_df = merged_df[merged_df.index > '2006-01-01']

X1 = merged_df[['SP500_quarterly_return', 'unemployment_rate', 'inflation', 'pce']]
X1 = sm.add_constant(X1)  
# Regression by party
y_all = merged_df['ics_all']
model = sm.OLS(y_all, X1, missing='drop').fit()
merged_df['res_all'] = model.resid

# Get the predictions
merged_df['predicted'] = model.predict(X1)

#Plot difference
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.figure(figsize=(14, 7))
plt.plot(merged_df.index, merged_df['ics_all'], color='darkgreen', label='Observed Sentiment')
plt.plot(merged_df.index, merged_df['predicted'], color='purple', linestyle='--', label='Predicted Sentiment')
plt.fill_between(merged_df.index, merged_df['ics_all'], merged_df['predicted'], color='coral', alpha=0.2)
plt.title('Actual vs. Predicted Consumer Sentiment from Baseline Model', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Consumer Sentiment', fontsize=14)
plt.legend(fontsize=14)
footnote = "Source: University of Michigan, Federal Reserve Bank of St. Louis, Yahoo! Finance, author calculations."
plt.figtext(0.1, -0.1, footnote, wrap=True, horizontalalignment='left', fontsize=14)
sns.despine()
plt.tight_layout()

# Show the plot
plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Regression 1-U3, CPI, PCE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
merged_df = merged_df[merged_df.index < '2020-03-01']

#Merge in the political sentiment dataframe
merged_df.drop('date', axis=1, inplace=True)
merged_df=merged_df.merge(pol_df, on='date', how='right')

# Set up controls and intercept
X1 = merged_df[['SP500_quarterly_return', 'unemployment_rate', 'inflation', 'pce']]
X1 = sm.add_constant(X1)  # Add a constant for the intercept

# Regression by party
y_dem = merged_df['ICS_Dem']
y_rep = merged_df['ICS_Rep']
y_ind = merged_df['ICS_Ind']
y_vars = [y_dem, y_rep, y_ind]
residual_names = ['residual_dem1', 'residual_rep1', 'residual_ind1']

for y_var, res_name in zip(y_vars, residual_names):
    model = sm.OLS(y_var, X1, missing='drop').fit()
    merged_df[res_name] = model.resid

# Plotting the residuals
plt.figure(figsize=(12, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
sns.lineplot(data=merged_df, x='date', y='residual_dem1', color='blue', label='Democrat')
sns.lineplot(data=merged_df, x='date', y='residual_rep1', color='red', label='Republican')

date_range = pd.date_range(start=merged_df['date'].min(), end=merged_df['date'].max())

# Create horizontal lines
presidency_periods = []
current_party = None
start_date = None

# Iterate over the date range and determine the presidency periods
for current_date in date_range:
    party = assign_party(current_date)
    if party != current_party:
        if current_party is not None:
            # End of a presidency period
            presidency_periods.append({'start': start_date, 'end': current_date, 'party': current_party})
        # Start of a new presidency period
        start_date = current_date
        current_party = party

presidency_periods.append({'start': start_date, 'end': date_range[-1], 'party': current_party})

# Convert start and end dates to matplotlib date format
for period in presidency_periods:
    period['start'] = mdates.date2num(period['start'])
    period['end'] = mdates.date2num(period['end'])


# Horizontal lines for residuals
for period in presidency_periods:
    if period['party'] == 'Republican':
        plt.hlines(y=15.02, xmin=period['start'], xmax=period['end'], colors='red', linewidth=2)
        plt.hlines(y=-6.2, xmin=period['start'], xmax=period['end'], colors='blue', linewidth=2)
    elif period['party'] == 'Democrat':
        plt.hlines(y=-14.26, xmin=period['start'], xmax=period['end'], colors='red', linewidth=2)
        plt.hlines(y=5.89, xmin=period['start'], xmax=period['end'], colors='blue', linewidth=2)


# Adding vertical lines for the specified dates
dates_to_highlight = [mdates.datestr2num('2008-12-01'),
                      mdates.datestr2num('2016-12-01'), mdates.datestr2num('2020-12-01')]
for date in dates_to_highlight:
    plt.axvline(x=date, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(y=0, color='black', linestyle='--')

plt.title("Residuals from Regressions on Consumer Sentiment by Party Affiliation", fontsize=16, fontweight='bold')
ax = plt.gca()  
# Set the y-label with horizontal orientation and move it above the axis
ax.set_ylabel('Residuals', fontsize=14, rotation=0, labelpad=20)
ax.yaxis.set_label_coords(0, 1.0) 
plt.xlabel("Date", fontsize=14)
sns.despine()
plt.legend()
source_note_text = "Source: Federal Reserve Bank of St. Louis, Yahoo!Finance, University of Michigan, author calculations. \nNote: Results are from a regression of Consumer Sentiment by party affiliation on the quarterly S&P 500 return, the unemployment rate (U3), annual CPI % change, and the level of Personal Consumption Expenditures (PCE)"
source_note = plt.figtext(0.1, -0.05, source_note_text, wrap=True, horizontalalignment='left', fontsize=12)
sns.despine()
plt.tight_layout()
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
         Regression 2-U3, Gas prices, Food prices, nominal wages, epop
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X2 = merged_df[['food_change', 'gas_change', 'nom_wages', 'epop']]
X2 = sm.add_constant(X2)  

# Regressions by party
y_vars = [y_dem, y_rep, y_ind]
residual_names_2 = ['residual_dem2', 'residual_rep2', 'residual_ind2']

for y_var, res_name in zip(y_vars, residual_names_2):
    model = sm.OLS(y_var, X2, missing='drop').fit()
    merged_df[res_name] = model.resid


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
         Regression 3-U3 and inflation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X3 = merged_df[['inflation', 'unemployment_rate']]
X3 = sm.add_constant(X3)  # Add a constant for the intercept

# Regression for ICS_Dem
y_vars = [y_dem, y_rep, y_ind]
residual_names_3 = ['residual_dem3', 'residual_rep3', 'residual_ind3']

for y_var, res_name in zip(y_vars, residual_names_3):
    model = sm.OLS(y_var, X3, missing='drop').fit()
    merged_df[res_name] = model.resid

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        Generate values for table (rest done in excel)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Calculate conditional average residuals
avg_residuals = {
    "Democrat": {
        "y_dem": [np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_dem1']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_dem2']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_dem3'])],
        "y_rep": [np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_rep1']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_rep2']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_rep3'])],
        "y_ind": [np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_ind1']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_ind2']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Democrat']['residual_ind3'])]
    },
    "Republican": {
        "y_dem": [np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_dem1']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_dem2']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_dem3'])],
        "y_rep": [np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_rep1']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_rep2']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_rep3'])],
        "y_ind": [np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_ind1']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_ind2']),
                  np.mean(merged_df[merged_df['presidential_party'] == 'Republican']['residual_ind3'])]
    }
}

# Put into dataframe
df_avg_residuals = pd.DataFrame(avg_residuals)

# Round to two decimals
for col in df_avg_residuals.columns:
    for index in df_avg_residuals.index:
        if isinstance(df_avg_residuals.at[index, col], list):  # Check if the value is a list
            # Round each element in the list to two decimal places
            df_avg_residuals.at[index, col] = [round(elem, 2) if isinstance(elem, float) else elem for elem in df_avg_residuals.at[index, col]]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        Adjusted consumer sentiment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Create summary stats
average_ics_rep = merged_df['ICS_Rep'].mean()
std_dev_ics_rep = merged_df['ICS_Rep'].std()
std_dev_ics_all = merged_df['ics_all'].std()
average_ics_ind = merged_df['ICS_Ind'].mean()
std_dev_ics_ind = merged_df['ICS_Ind'].std()
average_ics_dem = merged_df['ICS_Dem'].mean()
std_dev_ics_dem = merged_df['ICS_Dem'].std()

# Calculate scaled variables
merged_df['scaled_r_sent'] = ((merged_df['ICS_Rep'] - average_ics_rep) / (std_dev_ics_rep / std_dev_ics_all)) + average_ics_rep
merged_df['scaled_i_sent'] = ((merged_df['ICS_Ind'] - average_ics_ind) / (std_dev_ics_ind / std_dev_ics_all)) + average_ics_ind
merged_df['scaled_d_sent'] = ((merged_df['ICS_Dem'] - average_ics_dem) / (std_dev_ics_dem / std_dev_ics_all)) + average_ics_dem


###################### Chart showing adjusted vs. regular #####################
r_weight=0.286452947
d_weight=0.317476732
i_weight=0.396070321

# Calculate the adjusted sentiment
merged_df['adjusted_sentiment'] = (
    merged_df['scaled_r_sent'] * r_weight +
    merged_df['scaled_i_sent'] * i_weight ++
    merged_df['scaled_d_sent'] * d_weight
)



# Plotting observed vs. adjusted sentiment
plt.figure(figsize=(14, 7))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

plt.plot(merged_df['date'], merged_df['ics_all'], color='gray', label='Reported Sentiment')
plt.plot(merged_df['date'], merged_df['adjusted_sentiment'], 'k--', label='Adjusted Sentiment')

plt.legend(fontsize=14)
plt.title('Observed vs Adjusted Sentiment', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Sentiment', fontsize=14)
plt.tight_layout()
source_note_text = "Source: Author calculations. \nNote: To create the adjusted sentiment measure, for each political subindex, the adjusted index is = [(subindex value - subindex mean)/ (standard deviation of the subindex/overall consumer sentiment index standard deviation)]+subindex mean. The resulting overall adjusted index is a weighted average of the three adjusted political subindices, where the weights are rescaled such that those who listed their political affiliation as 'don't know' or 'N/A' are excluded"
source_note = plt.figtext(0.1, -0.05, source_note_text, wrap=True, horizontalalignment='left', fontsize=12)
plt.tight_layout()
sns.despine()
plt.show()

