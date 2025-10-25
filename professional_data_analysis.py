import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'sans-serif'

# Generate realistic e-commerce dataset
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2024-09-30', freq='D')
n_days = len(dates)

# Create synthetic e-commerce data
data = {
    'Date': dates,
    'Daily_Revenue': np.cumsum(np.random.normal(5000, 2000, n_days)) + np.linspace(0, 100000, n_days),
    'Transactions': np.random.poisson(150, n_days),
    'Conversion_Rate': np.random.uniform(2.0, 8.0, n_days),
    'Average_Order_Value': np.random.normal(85, 20, n_days),
    'Customer_Acquisition_Cost': np.random.uniform(15, 45, n_days),
    'Website_Traffic': np.random.poisson(5000, n_days),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_days),
    'Device_Type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_days),
}

df = pd.DataFrame(data)

# Calculate additional metrics
df['Month'] = df['Date'].dt.to_period('M')
df['Week'] = df['Date'].dt.to_period('W')
df['Day_of_Week'] = df['Date'].dt.day_name()
df['ROI'] = (df['Daily_Revenue'] / (df['Customer_Acquisition_Cost'] * df['Transactions'])) * 100

# Monthly aggregation
monthly_data = df.groupby('Month').agg({
    'Daily_Revenue': 'sum',
    'Transactions': 'sum',
    'Conversion_Rate': 'mean',
    'Average_Order_Value': 'mean',
    'Website_Traffic': 'sum',
    'ROI': 'mean'
}).reset_index()
monthly_data['Month'] = monthly_data['Month'].astype(str)

# Create comprehensive dashboard
fig = plt.figure(figsize=(12, 9))
fig.suptitle('E-Commerce Business Performance Dashboard - Detailed Analysis', 
             fontsize=14, fontweight='bold', y=0.995)

# 1. Revenue Trend with Moving Average
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df['Date'], df['Daily_Revenue'], color='#1f77b4', linewidth=1.5, alpha=0.7, label='Daily Revenue')
ma_30 = df['Daily_Revenue'].rolling(window=30).mean()
ax1.plot(df['Date'], ma_30, color='#ff7f0e', linewidth=2.5, label='30-Day Moving Average')
ax1.fill_between(df['Date'], df['Daily_Revenue'], alpha=0.2, color='#1f77b4')
ax1.set_title('Revenue Trend Analysis', fontsize=12, fontweight='bold')
ax1.set_ylabel('Revenue ($)', fontsize=10)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Monthly Revenue Comparison
ax2 = plt.subplot(3, 3, 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(monthly_data)))
bars = ax2.bar(range(len(monthly_data)), monthly_data['Daily_Revenue'], color=colors, edgecolor='black', linewidth=1.2)
ax2.set_title('Monthly Revenue Distribution', fontsize=12, fontweight='bold')
ax2.set_ylabel('Revenue ($)', fontsize=10)
ax2.set_xlabel('Month', fontsize=10)
ax2.set_xticks(range(len(monthly_data)))
ax2.set_xticklabels([m[:7] for m in monthly_data['Month']], rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'${height/1000:.0f}K', ha='center', va='bottom', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Conversion Rate Over Time
ax3 = plt.subplot(3, 3, 3)
scatter = ax3.scatter(df['Date'], df['Conversion_Rate'], c=df['Daily_Revenue'], 
                     cmap='plasma', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.plot(df['Date'], df['Conversion_Rate'].rolling(window=14).mean(), 
         color='red', linewidth=2, label='14-Day Trend', linestyle='--')
ax3.set_title('Conversion Rate Analysis', fontsize=12, fontweight='bold')
ax3.set_ylabel('Conversion Rate (%)', fontsize=10)
ax3.legend(fontsize=9)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Revenue ($)', fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Transaction Volume Distribution
ax4 = plt.subplot(3, 3, 4)
ax4.hist(df['Transactions'], bins=50, color='#2ca02c', edgecolor='black', alpha=0.7)
ax4.axvline(df['Transactions'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Transactions"].mean():.0f}')
ax4.axvline(df['Transactions'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["Transactions"].median():.0f}')
ax4.set_title('Daily Transaction Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Number of Transactions', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Average Order Value vs Website Traffic
ax5 = plt.subplot(3, 3, 5)
scatter2 = ax5.scatter(df['Website_Traffic'], df['Average_Order_Value'], 
                      c=df['Conversion_Rate'], cmap='coolwarm', s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['Website_Traffic'], df['Average_Order_Value'], 2)
p = np.poly1d(z)
x_trend = np.sort(df['Website_Traffic'])
ax5.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend Line')
ax5.set_title('Traffic vs Average Order Value', fontsize=12, fontweight='bold')
ax5.set_xlabel('Website Traffic (Sessions)', fontsize=10)
ax5.set_ylabel('Average Order Value ($)', fontsize=10)
ax5.legend(fontsize=9)
cbar2 = plt.colorbar(scatter2, ax=ax5)
cbar2.set_label('Conversion Rate (%)', fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Customer Acquisition Cost Trend
ax6 = plt.subplot(3, 3, 6)
ax6.fill_between(df['Date'], df['Customer_Acquisition_Cost'], alpha=0.3, color='#d62728')
ax6.plot(df['Date'], df['Customer_Acquisition_Cost'], color='#d62728', linewidth=2)
ax6.plot(df['Date'], df['Customer_Acquisition_Cost'].rolling(window=30).mean(), 
         color='#1f77b4', linewidth=2.5, label='30-Day Average')
ax6.set_title('Customer Acquisition Cost Trend', fontsize=12, fontweight='bold')
ax6.set_ylabel('CAC ($)', fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# 7. Revenue by Product Category (Pie Chart)
ax7 = plt.subplot(3, 3, 7)
category_revenue = df.groupby('Product_Category')['Daily_Revenue'].sum()
colors_pie = plt.cm.Set3(range(len(category_revenue)))
wedges, texts, autotexts = ax7.pie(category_revenue, labels=category_revenue.index, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90, textprops={'fontsize': 9})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax7.set_title('Revenue by Product Category', fontsize=12, fontweight='bold')

# 8. Device Type Performance
ax8 = plt.subplot(3, 3, 8)
device_stats = df.groupby('Device_Type').agg({
    'Daily_Revenue': 'sum',
    'Transactions': 'sum',
    'Conversion_Rate': 'mean'
}).reset_index()
x_pos = np.arange(len(device_stats))
width = 0.35
ax8_twin = ax8.twinx()

bars1 = ax8.bar(x_pos - width/2, device_stats['Daily_Revenue']/1000, width, 
                label='Revenue ($K)', color='#1f77b4', edgecolor='black', linewidth=1.2)
bars2 = ax8_twin.bar(x_pos + width/2, device_stats['Conversion_Rate'], width, 
                     label='Conv. Rate (%)', color='#ff7f0e', edgecolor='black', linewidth=1.2)

ax8.set_title('Performance by Device Type', fontsize=12, fontweight='bold')
ax8.set_ylabel('Revenue ($K)', fontsize=10, color='#1f77b4')
ax8_twin.set_ylabel('Conversion Rate (%)', fontsize=10, color='#ff7f0e')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(device_stats['Device_Type'])
ax8.tick_params(axis='y', labelcolor='#1f77b4')
ax8_twin.tick_params(axis='y', labelcolor='#ff7f0e')
ax8.grid(True, alpha=0.3, axis='y')

# 9. ROI Analysis (Heatmap style)
ax9 = plt.subplot(3, 3, 9)
weekly_roi = df.groupby('Week').agg({
    'ROI': 'mean',
    'Daily_Revenue': 'sum',
    'Transactions': 'sum'
}).reset_index()
weekly_roi = weekly_roi.tail(20)
scatter3 = ax9.scatter(range(len(weekly_roi)), weekly_roi['ROI'], 
                      s=weekly_roi['Daily_Revenue']/100, 
                      c=weekly_roi['ROI'], cmap='RdYlGn', 
                      alpha=0.6, edgecolors='black', linewidth=1)
ax9.plot(range(len(weekly_roi)), weekly_roi['ROI'], color='navy', linewidth=2, alpha=0.5)
ax9.set_title('Weekly ROI Analysis (Last 20 Weeks)', fontsize=12, fontweight='bold')
ax9.set_ylabel('ROI (%)', fontsize=10)
ax9.set_xlabel('Week', fontsize=10)
ax9.grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=ax9)
cbar3.set_label('ROI (%)', fontsize=9)

plt.tight_layout()
plt.savefig('dashboard_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'dashboard_analysis.png'")

# Generate Key Performance Indicators Report
print("\n" + "="*80)
print("KEY PERFORMANCE INDICATORS - EXECUTIVE SUMMARY")
print("="*80)
print(f"\nAnalysis Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Total Period: {(df['Date'].max() - df['Date'].min()).days} days")
print(f"\n{'Metric':<35} {'Value':<20} {'Change':<15}")
print("-"*70)
print(f"{'Total Revenue':<35} ${df['Daily_Revenue'].sum():>18,.2f} {'':>15}")
print(f"{'Average Daily Revenue':<35} ${df['Daily_Revenue'].mean():>18,.2f} {'':>15}")
print(f"{'Total Transactions':<35} {df['Transactions'].sum():>19,.0f} {'':>15}")
print(f"{'Average Conversion Rate':<35} {df['Conversion_Rate'].mean():>18.2f}% {'':>15}")
print(f"{'Average Order Value':<35} ${df['Average_Order_Value'].mean():>18,.2f} {'':>15}")
print(f"{'Average Customer Acquisition Cost':<35} ${df['Customer_Acquisition_Cost'].mean():>18,.2f} {'':>15}")
print(f"{'Total Website Traffic':<35} {df['Website_Traffic'].sum():>19,.0f} {'':>15}")
print(f"{'Average ROI':<35} {df['ROI'].mean():>18.2f}% {'':>15}")

# Growth Analysis
first_month_revenue = monthly_data['Daily_Revenue'].iloc[0]
last_month_revenue = monthly_data['Daily_Revenue'].iloc[-1]
growth_rate = ((last_month_revenue - first_month_revenue) / first_month_revenue) * 100

print(f"\n{'Growth Metrics':<35}")
print("-"*70)
print(f"{'Month-over-Month Growth':<35} {growth_rate:>18.2f}% {'':>15}")
print(f"{'Best Performing Month':<35} {monthly_data.loc[monthly_data['Daily_Revenue'].idxmax(), 'Month']:>18} {'':>15}")
print(f"{'Lowest Performing Month':<35} {monthly_data.loc[monthly_data['Daily_Revenue'].idxmin(), 'Month']:>18} {'':>15}")

# Category Performance
print(f"\n{'Category Performance':<35}")
print("-"*70)
category_perf = df.groupby('Product_Category').agg({
    'Daily_Revenue': 'sum',
    'Transactions': 'sum',
    'Conversion_Rate': 'mean'
}).sort_values('Daily_Revenue', ascending=False)

for category in category_perf.index[:3]:
    revenue = category_perf.loc[category, 'Daily_Revenue']
    pct = (revenue / df['Daily_Revenue'].sum()) * 100
    print(f"{'  ' + category:<35} ${revenue:>18,.2f} ({pct:>5.1f}%)")

print("\n" + "="*80)

# Create a second detailed analysis figure
fig2 = plt.figure(figsize=(12, 8))
fig2.suptitle('Advanced Statistical Analysis & Correlations', fontsize=12, fontweight='bold', y=0.995)

# 1. Correlation Heatmap
ax10 = plt.subplot(2, 3, 1)
numeric_cols = ['Daily_Revenue', 'Transactions', 'Conversion_Rate', 
                'Average_Order_Value', 'Customer_Acquisition_Cost', 'Website_Traffic', 'ROI']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=ax10, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
ax10.set_title('Correlation Matrix Heatmap', fontsize=12, fontweight='bold')

# 2. Box Plot - Revenue Distribution by Category
ax11 = plt.subplot(2, 3, 2)
df.boxplot(column='Daily_Revenue', by='Product_Category', ax=ax11)
ax11.set_title('Revenue Distribution by Category', fontsize=12, fontweight='bold')
ax11.set_xlabel('Product Category', fontsize=10)
ax11.set_ylabel('Daily Revenue ($)', fontsize=10)
plt.sca(ax11)
plt.xticks(rotation=45, ha='right')

# 3. Day of Week Performance
ax12 = plt.subplot(2, 3, 3)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_perf = df.groupby('Day_of_Week')['Daily_Revenue'].agg(['mean', 'std']).reindex(day_order)
ax12.bar(range(len(day_perf)), day_perf['mean'], yerr=day_perf['std'], 
         color='#2ca02c', edgecolor='black', linewidth=1.2, capsize=5, alpha=0.7)
ax12.set_title('Average Daily Revenue by Day of Week', fontsize=12, fontweight='bold')
ax12.set_ylabel('Average Revenue ($)', fontsize=10)
ax12.set_xticks(range(len(day_perf)))
ax12.set_xticklabels([d[:3] for d in day_perf.index], rotation=45, ha='right')
ax12.grid(True, alpha=0.3, axis='y')

# 4. Monthly Trend Lines Comparison
ax13 = plt.subplot(2, 3, 4)
for category in df['Product_Category'].unique():
    category_data = df[df['Product_Category'] == category].groupby('Month')['Daily_Revenue'].sum()
    ax13.plot(range(len(category_data)), category_data.values, marker='o', label=category, linewidth=2)
ax13.set_title('Monthly Revenue Trends by Category', fontsize=12, fontweight='bold')
ax13.set_ylabel('Revenue ($)', fontsize=10)
ax13.set_xlabel('Month', fontsize=10)
ax13.legend(fontsize=9, loc='best')
ax13.grid(True, alpha=0.3)

# 5. Violin Plot - Conversion Rate Distribution
ax14 = plt.subplot(2, 3, 5)
parts = ax14.violinplot([df[df['Product_Category'] == cat]['Conversion_Rate'].values 
                         for cat in df['Product_Category'].unique()],
                        positions=range(len(df['Product_Category'].unique())), widths=0.7,
                        showmeans=True, showmedians=True)
ax14.set_title('Conversion Rate Distribution (Violin Plot)', fontsize=12, fontweight='bold')
ax14.set_ylabel('Conversion Rate (%)', fontsize=10)
ax14.set_xticks(range(len(df['Product_Category'].unique())))
ax14.set_xticklabels(df['Product_Category'].unique(), rotation=45, ha='right')
ax14.grid(True, alpha=0.3, axis='y')

# 6. Funnel Analysis (Waterfall style)
ax15 = plt.subplot(2, 3, 6)
funnel_stages = ['Website_Traffic', 'Transactions', 'Revenue']
funnel_values = [
    df['Website_Traffic'].sum() / 1000,  # in thousands for readability
    df['Transactions'].sum() / 10,        # scaled
    df['Daily_Revenue'].sum() / 100000    # in hundreds of thousands
]
colors_funnel = ['#1f77b4', '#ff7f0e', '#2ca02c']
ax15.barh(funnel_stages, funnel_values, color=colors_funnel, edgecolor='black', linewidth=1.5)
for i, v in enumerate(funnel_values):
    ax15.text(v + max(funnel_values)*0.02, i, f'{v:.1f}', va='center', fontweight='bold')
ax15.set_title('Conversion Funnel Analysis', fontsize=12, fontweight='bold')
ax15.set_xlabel('Scale (Traffic in K, Transactions /10, Revenue in $100K)', fontsize=9)
ax15.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('statistical_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Statistical analysis saved as 'statistical_analysis.png'")

# Save processed data
df.to_csv('ecommerce_data_processed.csv', index=False)
print("✓ Processed data saved as 'ecommerce_data_processed.csv'")

print("\n✓ Analysis complete! All visualizations have been generated successfully.")
