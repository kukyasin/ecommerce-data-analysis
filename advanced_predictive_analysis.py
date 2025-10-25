import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the previously generated data
df = pd.read_csv('ecommerce_data_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Create advanced analysis visualizations
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Advanced Predictive Analytics & Machine Learning Insights', 
             fontsize=14, fontweight='bold', y=0.995)

# 1. Time Series Decomposition - REVENUE
ax1 = plt.subplot(3, 3, 1)
revenue_7d = df['Daily_Revenue'].rolling(window=7).mean()
revenue_30d = df['Daily_Revenue'].rolling(window=30).mean()
trend = revenue_30d
seasonal = df['Daily_Revenue'] - revenue_7d
residual = df['Daily_Revenue'] - trend

ax1.plot(df['Date'], df['Daily_Revenue'], label='Original', alpha=0.6, linewidth=1)
ax1.plot(df['Date'], trend, label='Trend (30-day MA)', color='red', linewidth=2.5)
ax1.fill_between(df['Date'], trend, alpha=0.2, color='red')
ax1.set_title('Time Series Decomposition - Revenue Trend', fontsize=12, fontweight='bold')
ax1.set_ylabel('Revenue ($)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Growth Rate Analysis
ax2 = plt.subplot(3, 3, 2)
df['Daily_Revenue_pct_change'] = df['Daily_Revenue'].pct_change() * 100
growth_30d = df['Daily_Revenue_pct_change'].rolling(window=30).mean()
ax2.bar(df['Date'], df['Daily_Revenue_pct_change'], alpha=0.4, color='steelblue', label='Daily Growth %')
ax2.plot(df['Date'], growth_30d, color='darkred', linewidth=2.5, label='30-Day Growth Trend')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_title('Daily Revenue Growth Rate Analysis', fontsize=12, fontweight='bold')
ax2.set_ylabel('Growth Rate (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Cumulative Revenue & Profit Margin
ax3 = plt.subplot(3, 3, 3)
df['Cumulative_Revenue'] = df['Daily_Revenue'].cumsum()
df['Profit_Margin'] = (df['Daily_Revenue'] / df['Website_Traffic'] * df['Conversion_Rate']) * 100
ax3_twin = ax3.twinx()

line1 = ax3.plot(df['Date'], df['Cumulative_Revenue']/1e6, color='#1f77b4', linewidth=2.5, label='Cumulative Revenue')
line2 = ax3_twin.plot(df['Date'], df['Profit_Margin'], color='#ff7f0e', linewidth=2, label='Profit Margin %', linestyle='--')

ax3.set_title('Cumulative Revenue & Profit Margin', fontsize=12, fontweight='bold')
ax3.set_ylabel('Cumulative Revenue ($M)', fontsize=10, color='#1f77b4')
ax3_twin.set_ylabel('Profit Margin (%)', fontsize=10, color='#ff7f0e')
ax3.tick_params(axis='y', labelcolor='#1f77b4')
ax3_twin.tick_params(axis='y', labelcolor='#ff7f0e')
ax3.grid(True, alpha=0.3)

# 4. Statistical Distribution - Kernel Density Estimation
ax4 = plt.subplot(3, 3, 4)
for device in df['Device_Type'].unique():
    data = df[df['Device_Type'] == device]['Daily_Revenue']
    data.plot.kde(ax=ax4, linewidth=2.5, label=device)
ax4.set_title('Revenue Distribution by Device Type (KDE)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Daily Revenue ($)', fontsize=10)
ax4.set_ylabel('Density', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Quantile-Quantile Plot (Q-Q Plot) for Normality Test
ax5 = plt.subplot(3, 3, 5)
revenue_sample = df['Daily_Revenue'].sample(min(1000, len(df)), random_state=42)
stats.probplot(revenue_sample, dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot: Revenue Normality Assessment', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Moving Average Convergence Divergence (MACD)
ax6 = plt.subplot(3, 3, 6)
ema_12 = df['Daily_Revenue'].ewm(span=12).mean()
ema_26 = df['Daily_Revenue'].ewm(span=26).mean()
macd = ema_12 - ema_26
signal = macd.ewm(span=9).mean()
histogram = macd - signal

ax6.plot(df['Date'], macd, label='MACD', linewidth=2, color='blue')
ax6.plot(df['Date'], signal, label='Signal Line', linewidth=2, color='red', linestyle='--')
ax6.bar(df['Date'], histogram, label='Histogram', alpha=0.3, color='gray', width=1)
ax6.set_title('MACD (Moving Average Convergence Divergence)', fontsize=12, fontweight='bold')
ax6.set_ylabel('MACD Value', fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# 7. Volatility Analysis (Rolling Standard Deviation)
ax7 = plt.subplot(3, 3, 7)
volatility_30d = df['Daily_Revenue'].rolling(window=30).std()
ax7.fill_between(df['Date'], volatility_30d, alpha=0.3, color='purple')
ax7.plot(df['Date'], volatility_30d, color='purple', linewidth=2.5)
ax7.axhline(y=volatility_30d.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${volatility_30d.mean():,.0f}')
ax7.set_title('30-Day Revenue Volatility', fontsize=12, fontweight='bold')
ax7.set_ylabel('Standard Deviation ($)', fontsize=10)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# 8. Anomaly Detection - Z-Score Analysis
ax8 = plt.subplot(3, 3, 8)
z_scores = np.abs(stats.zscore(df['Daily_Revenue']))
anomalies = z_scores > 2.5
ax8.scatter(df['Date'][~anomalies], df['Daily_Revenue'][~anomalies], alpha=0.5, s=30, label='Normal', color='blue')
ax8.scatter(df['Date'][anomalies], df['Daily_Revenue'][anomalies], alpha=0.8, s=80, label='Anomaly', color='red', marker='X')
ax8.set_title('Anomaly Detection (Z-Score > 2.5)', fontsize=12, fontweight='bold')
ax8.set_ylabel('Daily Revenue ($)', fontsize=10)
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# 9. Sharpe Ratio Analysis (Risk-Adjusted Returns)
ax9 = plt.subplot(3, 3, 9)
returns = df['Daily_Revenue'].pct_change().dropna()
rolling_returns = returns.rolling(window=30).mean()
rolling_std = returns.rolling(window=30).std()
risk_free_rate = 0.02 / 252  # Annual 2% converted to daily
sharpe_ratio = (rolling_returns - risk_free_rate) / rolling_std

ax9.fill_between(df['Date'][1:].values, sharpe_ratio, alpha=0.3, color='green')
ax9.plot(df['Date'][1:].values, sharpe_ratio, color='darkgreen', linewidth=2.5, label='Sharpe Ratio')
ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax9.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=12, fontweight='bold')
ax9.set_ylabel('Sharpe Ratio', fontsize=10)
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Predictive analysis saved as 'predictive_analysis.png'")

# Create machine learning insights figure
fig2 = plt.figure(figsize=(12, 8))
fig2.suptitle('Machine Learning & Dimensionality Reduction Analysis', 
              fontsize=12, fontweight='bold', y=0.995)

# Prepare data for ML analysis
ml_features = ['Transactions', 'Conversion_Rate', 'Average_Order_Value', 
               'Customer_Acquisition_Cost', 'Website_Traffic', 'ROI']
X = df[ml_features].fillna(df[ml_features].mean()).values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA Analysis
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# 1. Scree Plot
ax10 = plt.subplot(2, 3, 1)
cumsum_var = np.cumsum(explained_variance)
ax10.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-', linewidth=2, markersize=8)
ax10.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.3, color='blue')
ax10.set_title('Scree Plot - Variance Explained by Components', fontsize=12, fontweight='bold')
ax10.set_xlabel('Principal Component', fontsize=10)
ax10.set_ylabel('Variance Explained', fontsize=10)
ax10.grid(True, alpha=0.3)

# 2. Cumulative Variance Explained
ax11 = plt.subplot(2, 3, 2)
ax11.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-', linewidth=2.5, markersize=8, label='Cumulative')
ax11.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Threshold')
ax11.fill_between(range(1, len(cumsum_var) + 1), cumsum_var, alpha=0.2, color='red')
ax11.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax11.set_xlabel('Number of Components', fontsize=10)
ax11.set_ylabel('Cumulative Variance', fontsize=10)
ax11.legend(fontsize=9)
ax11.grid(True, alpha=0.3)

# 3. Biplot (PC1 vs PC2)
ax12 = plt.subplot(2, 3, 3)
scatter = ax12.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Daily_Revenue'], cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
for i, feature in enumerate(ml_features):
    ax12.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3, 
              head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    ax12.text(pca.components_[0, i]*3.5, pca.components_[1, i]*3.5, feature, 
             fontsize=9, fontweight='bold', ha='center')
ax12.set_title('PCA Biplot (PC1 vs PC2)', fontsize=12, fontweight='bold')
ax12.set_xlabel(f'PC1 ({explained_variance[0]:.1%})', fontsize=10)
ax12.set_ylabel(f'PC2 ({explained_variance[1]:.1%})', fontsize=10)
cbar = plt.colorbar(scatter, ax=ax12)
cbar.set_label('Revenue ($)', fontsize=9)
ax12.grid(True, alpha=0.3)

# 4. Feature Importance (based on PCA loadings)
ax13 = plt.subplot(2, 3, 4)
loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
loading_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=ml_features)
loading_df.plot(kind='barh', ax=ax13, width=0.7, color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=1)
ax13.set_title('Feature Loadings on First Two PCs', fontsize=12, fontweight='bold')
ax13.set_xlabel('Loading Value', fontsize=10)
ax13.grid(True, alpha=0.3, axis='x')

# 5. Revenue Prediction - Simple Linear Regression Fit
ax14 = plt.subplot(2, 3, 5)
from sklearn.linear_model import LinearRegression
X_days = np.arange(len(df)).reshape(-1, 1)
model = LinearRegression()
model.fit(X_days, df['Daily_Revenue'].values)
y_pred = model.predict(X_days)
residuals = df['Daily_Revenue'].values - y_pred

ax14.scatter(df['Date'], residuals, alpha=0.5, s=30, color='blue')
ax14.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax14.fill_between(df['Date'], residuals, alpha=0.2, color='blue')
ax14.set_title('Linear Regression Residuals', fontsize=12, fontweight='bold')
ax14.set_ylabel('Residuals ($)', fontsize=10)
ax14.grid(True, alpha=0.3)

# 6. Distribution of Residuals
ax15 = plt.subplot(2, 3, 6)
ax15.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax15.axvline(x=residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${residuals.mean():,.0f}')
ax15.axvline(x=np.median(residuals), color='orange', linestyle='--', linewidth=2, label=f'Median: ${np.median(residuals):,.0f}')
stats_text = f"Std Dev: ${residuals.std():,.0f}\nSkewness: {stats.skew(residuals):.2f}"
ax15.text(0.98, 0.97, stats_text, transform=ax15.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax15.set_title('Residuals Distribution Analysis', fontsize=12, fontweight='bold')
ax15.set_xlabel('Residual Value ($)', fontsize=10)
ax15.set_ylabel('Frequency', fontsize=10)
ax15.legend(fontsize=9, loc='upper left')
ax15.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ml_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Machine Learning analysis saved as 'ml_analysis.png'")

# Generate detailed statistical report
print("\n" + "="*80)
print("ADVANCED STATISTICAL ANALYSIS REPORT")
print("="*80)

print("\n1. REVENUE STATISTICS")
print("-"*80)
print(f"Mean Revenue: ${df['Daily_Revenue'].mean():,.2f}")
print(f"Median Revenue: ${df['Daily_Revenue'].median():,.2f}")
print(f"Standard Deviation: ${df['Daily_Revenue'].std():,.2f}")
print(f"Coefficient of Variation: {(df['Daily_Revenue'].std() / df['Daily_Revenue'].mean()):.2%}")
print(f"Skewness: {stats.skew(df['Daily_Revenue']):.4f}")
print(f"Kurtosis: {stats.kurtosis(df['Daily_Revenue']):.4f}")
print(f"Min Revenue: ${df['Daily_Revenue'].min():,.2f}")
print(f"Max Revenue: ${df['Daily_Revenue'].max():,.2f}")
print(f"25th Percentile: ${df['Daily_Revenue'].quantile(0.25):,.2f}")
print(f"75th Percentile: ${df['Daily_Revenue'].quantile(0.75):,.2f}")
print(f"Interquartile Range: ${df['Daily_Revenue'].quantile(0.75) - df['Daily_Revenue'].quantile(0.25):,.2f}")

print("\n2. GROWTH ANALYSIS")
print("-"*80)
revenue_pct_changes = df['Daily_Revenue'].pct_change().dropna()
print(f"Average Daily Growth: {revenue_pct_changes.mean():.4%}")
print(f"Daily Growth Std Dev: {revenue_pct_changes.std():.4%}")
print(f"Best Day Growth: {revenue_pct_changes.max():.2%}")
print(f"Worst Day Growth: {revenue_pct_changes.min():.2%}")

print("\n3. VOLATILITY METRICS")
print("-"*80)
vol_30d = df['Daily_Revenue'].rolling(window=30).std()
print(f"Average 30-Day Volatility: ${vol_30d.mean():,.2f}")
print(f"Min Volatility: ${vol_30d.min():,.2f}")
print(f"Max Volatility: ${vol_30d.max():,.2f}")
annualized_vol = df['Daily_Revenue'].pct_change().std() * np.sqrt(365)
print(f"Annualized Volatility: {annualized_vol:.2%}")

print("\n4. TREND ANALYSIS")
print("-"*80)
slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(df)), df['Daily_Revenue'].values)
print(f"Trend Slope: ${slope:,.2f} per day")
print(f"Linear R² Value: {r_value**2:.4f}")
print(f"Trend Statistical Significance (p-value): {p_value:.2e}")
projected_revenue_90days = df['Daily_Revenue'].iloc[-1] + (slope * 90)
print(f"Projected Revenue (90 days): ${projected_revenue_90days:,.2f}")

print("\n5. CORRELATION ANALYSIS")
print("-"*80)
correlations = df[ml_features + ['Daily_Revenue']].corr()['Daily_Revenue'].sort_values(ascending=False)
for feature, corr in correlations.items():
    print(f"{feature:30s}: {corr:7.4f}")

print("\n6. ANOMALY DETECTION")
print("-"*80)
z_scores = np.abs(stats.zscore(df['Daily_Revenue']))
anomalies_count = (z_scores > 2.5).sum()
anomalies_pct = (anomalies_count / len(df)) * 100
print(f"Detected Anomalies (Z-score > 2.5): {anomalies_count} ({anomalies_pct:.2f}%)")
print(f"Anomaly Dates:")
anomaly_dates = df[z_scores > 2.5][['Date', 'Daily_Revenue']].head(5)
for idx, row in anomaly_dates.iterrows():
    print(f"  {row['Date'].date()}: ${row['Daily_Revenue']:,.2f}")

print("\n7. MODEL PERFORMANCE")
print("-"*80)
r_squared = 1 - (np.sum(residuals**2) / np.sum((df['Daily_Revenue'].values - df['Daily_Revenue'].mean())**2))
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
print(f"Linear Model R² Score: {r_squared:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {np.mean(np.abs(residuals / df['Daily_Revenue'].values)):.2%}")

print("\n8. DIMENSIONALITY REDUCTION (PCA)")
print("-"*80)
print("Variance Explained by Each Component:")
for i, var in enumerate(explained_variance[:4]):
    print(f"  PC{i+1}: {var:.2%}")
cumsum_95 = np.where(np.cumsum(explained_variance) >= 0.95)[0][0] + 1
print(f"Components needed for 95% variance: {cumsum_95}")

print("\n" + "="*80)
print("✓ All advanced analyses completed successfully!")
print("="*80 + "\n")
