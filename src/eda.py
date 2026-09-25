import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 13

def univariate_analysis(df_sales, df_prices, output_dir="reports/figures"):
    """Performs univariate analysis on unit sales distribution and product pricing."""
    print("\n[EDA] === Running 1. Univariate Analysis ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    all_sales_sample = df_sales[date_cols].values.flatten()
    sample_subset = np.random.choice(all_sales_sample, size=min(100000, len(all_sales_sample)), replace=False)
    
    zero_ratio = np.mean(sample_subset == 0) * 100
    non_zero_sales = sample_subset[sample_subset > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].pie(
        [zero_ratio, 100 - zero_ratio],
        labels=[f'Zero Sales ({zero_ratio:.1f}%)', f'Positive Sales ({100-zero_ratio:.1f}%)'],
        autopct='%1.1f%%',
        colors=['#ff7f0e', '#1f77b4'],
        startangle=140,
        explode=(0.08, 0)
    )
    axes[0].set_title("Univariate: Zero Sales Proportion (Intermittent Demand)", fontweight='bold')
    
    sns.histplot(non_zero_sales, bins=40, kde=True, ax=axes[1], color='#2ca02c')
    axes[1].set_yscale('log')
    axes[1].set_title("Univariate: Non-Zero Sales Frequency Distribution (Log Scale)", fontweight='bold')
    axes[1].set_xlabel("Daily Units Sold per Item")
    axes[1].set_ylabel("Log Frequency")
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_1_univariate_sales_distribution.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Sales distribution plot saved to: {p1}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df_prices_with_cat = df_prices.copy()
    df_prices_with_cat['category'] = df_prices_with_cat['item_id'].apply(lambda x: x.split('_')[0])
    
    sns.boxplot(data=df_prices_with_cat, x='category', y='sell_price', hue='category', palette='Set2', ax=ax, showfliers=False, legend=False)
    ax.set_title("Univariate: Product Price Spread across Categories ($ USD)", fontweight='bold')
    ax.set_ylabel("Sell Price ($)")
    ax.set_xlabel("Category")
    
    plt.tight_layout()
    p2 = os.path.join(output_dir, "eda_1_univariate_price_distribution.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"   [SAVED] Category price distribution plot saved to: {p2}")

def bivariate_analysis(df_sales, df_calendar, df_prices, output_dir="reports/figures"):
    """Performs bivariate analysis on store, state, category and SNAP interactions."""
    print("\n[EDA] === Running 2. Bivariate Analysis ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    
    df_sales_store = df_sales.groupby(['state_id', 'store_id'])[date_cols].sum().sum(axis=1).reset_index()
    df_sales_store.columns = ['state_id', 'store_id', 'total_sales']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    state_sales = df_sales_store.groupby('state_id')['total_sales'].sum().reset_index()
    sns.barplot(data=state_sales, x='state_id', y='total_sales', hue='state_id', palette='Blues_r', ax=axes[0], legend=False)
    axes[0].set_title("Bivariate: Total Sales Volume by State", fontweight='bold')
    axes[0].set_ylabel("Total Units Sold")
    axes[0].set_xlabel("State")
    
    sns.barplot(data=df_sales_store.sort_values('total_sales', ascending=False), 
                x='store_id', y='total_sales', hue='state_id', dodge=False, palette='tab10', ax=axes[1])
    axes[1].set_title("Bivariate: Total Sales Volume by Store (CA_3 Top Performer)", fontweight='bold')
    axes[1].set_ylabel("Total Units Sold")
    axes[1].set_xlabel("Store ID")
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_2_bivariate_store_state_sales.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Store/State sales comparison saved to: {p1}")
    
    cat_sales = df_sales.groupby('cat_id')[date_cols].sum().T
    cat_sales.index = df_calendar.set_index('d').loc[cat_sales.index, 'date']
    cat_sales.index = pd.to_datetime(cat_sales.index)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    for cat in cat_sales.columns:
        cat_sales[cat].rolling(30).mean().plot(ax=ax, label=f"{cat} (30-day MA)", linewidth=2)
        
    ax.set_title("Bivariate: Category Sales Velocity Trajectories (30-Day Moving Average)", fontweight='bold')
    ax.set_ylabel("Daily Units Sold")
    ax.set_xlabel("Date")
    ax.legend()
    
    plt.tight_layout()
    p2 = os.path.join(output_dir, "eda_2_bivariate_category_trends.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"   [SAVED] Category sales trajectories saved to: {p2}")
    
    ca_foods = df_sales[(df_sales['state_id'] == 'CA') & (df_sales['cat_id'] == 'FOODS')][date_cols].sum()
    ca_non_foods = df_sales[(df_sales['state_id'] == 'CA') & (df_sales['cat_id'] != 'FOODS')][date_cols].sum()
    
    df_snap_ca = pd.DataFrame({
        'd': date_cols,
        'foods_sales': ca_foods.values,
        'non_foods_sales': ca_non_foods.values
    }).merge(df_calendar[['d', 'snap_CA']], on='d')
    
    snap_comparison = df_snap_ca.groupby('snap_CA')[['foods_sales', 'non_foods_sales']].mean().reset_index()
    snap_comparison['snap_CA'] = snap_comparison['snap_CA'].map({0: 'Non-SNAP Days', 1: 'SNAP Active Days'})
    
    fig, ax = plt.subplots(figsize=(8, 5))
    df_snap_melt = snap_comparison.melt(id_vars='snap_CA', value_vars=['foods_sales', 'non_foods_sales'],
                                        var_name='Department', value_name='Average Daily Sales')
    df_snap_melt['Department'] = df_snap_melt['Department'].map({'foods_sales': 'FOODS', 'non_foods_sales': 'NON-FOODS'})
    
    sns.barplot(data=df_snap_melt, x='Department', y='Average Daily Sales', hue='snap_CA', palette='Set1', ax=ax)
    ax.set_title("Bivariate: Policy Impact of SNAP Days on California Sales", fontweight='bold')
    ax.set_ylabel("Mean Daily Units Sold")
    
    plt.tight_layout()
    p3 = os.path.join(output_dir, "eda_2_bivariate_snap_impact.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"   [SAVED] SNAP impact policy comparison saved to: {p3}")

def multivariate_time_series_analysis(df_sales, df_calendar, output_dir="reports/figures"):
    """Performs multivariate time series decomposition and autocorrelation analysis."""
    print("\n[EDA] === Running 3. Multivariate & Time Series Decomposition ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    
    total_daily = df_sales[date_cols].sum()
    df_ts = pd.DataFrame({
        'd': date_cols,
        'total_sales': total_daily.values
    }).merge(df_calendar[['d', 'date', 'weekday', 'wday', 'month', 'year']], on='d')
    
    df_ts['date'] = pd.to_datetime(df_ts['date'])
    df_ts = df_ts.sort_values('date').set_index('date')
    df_ts['rolling_7'] = df_ts['total_sales'].rolling(7).mean()
    df_ts['rolling_30'] = df_ts['total_sales'].rolling(30).mean()
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_ts.index, df_ts['total_sales'], label='Raw Daily Sales', alpha=0.35, color='gray')
    ax.plot(df_ts.index, df_ts['rolling_7'], label='7-Day Moving Avg (Weekly Trend)', color='#1f77b4', linewidth=1.5)
    ax.plot(df_ts.index, df_ts['rolling_30'], label='30-Day Moving Avg (Monthly Trend)', color='#d62728', linewidth=2.5)
    
    xmas_dates = df_ts[df_ts.index.month == 12]
    xmas_days = xmas_dates[xmas_dates.index.day == 25]
    for xmas in xmas_days.index:
        ax.axvline(xmas, color='red', linestyle='--', alpha=0.5)
        ax.text(xmas, 5000, 'Christmas Shutdown', rotation=90, color='red', fontsize=9, ha='right')
        
    ax.set_title("Multivariate: Walmart Macro Sales Trajectory (2011 - 2016) with Holiday Shock Signals", fontweight='bold')
    ax.set_ylabel("Total Units Sold Across All Stores")
    ax.set_xlabel("Year")
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_3_multivariate_macro_trend.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Macro Trend plot saved to: {p1}")
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cat_daily = df_sales.groupby('cat_id')[date_cols].sum().T.reset_index()
    cat_daily = cat_daily.merge(df_calendar[['d', 'weekday']], left_on='index', right_on='d')
    cat_wday = cat_daily.groupby('weekday')[['FOODS', 'HOBBIES', 'HOUSEHOLD']].mean().loc[weekday_order]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cat_wday.T, cmap='YlGnBu', annot=True, fmt='.0f', cbar_kws={'label': 'Mean Daily Sales'}, ax=ax)
    ax.set_title("Multivariate: Weekly Seasonality Heatmap by Product Category", fontweight='bold')
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Category")
    
    plt.tight_layout()
    p2 = os.path.join(output_dir, "eda_3_multivariate_weekly_heatmap.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"   [SAVED] Seasonality Heatmap saved to: {p2}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(df_ts['total_sales'], lags=60, ax=axes[0], title="Autocorrelation (ACF) - Lags 1 to 60")
    plot_pacf(df_ts['total_sales'], lags=60, ax=axes[1], title="Partial Autocorrelation (PACF) - Lags 1 to 60", method='ywm')
    
    axes[0].axvline(7, color='green', linestyle=':', label='7-Day Cycle')
    axes[0].axvline(14, color='green', linestyle=':', label='14-Day Cycle')
    axes[0].axvline(28, color='orange', linestyle=':', label='28-Day Forecast Horizon')
    axes[0].legend()
    axes[1].axvline(7, color='green', linestyle=':', label='7-Day Cycle')
    axes[1].axvline(28, color='orange', linestyle=':', label='28-Day Horizon')
    axes[1].legend()
    
    plt.tight_layout()
    p3 = os.path.join(output_dir, "eda_3_multivariate_acf_pacf.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"   [SAVED] ACF/PACF correlation plots saved to: {p3}")

def composite_price_elasticity_analysis(df_sales, df_calendar, df_prices, output_dir="reports/figures"):
    """Constructs composite price features and analyzes price elasticity tiers."""
    print("\n[EDA] === Running 4. Composite Price Elasticity & Markdown Analysis ===")
    os.makedirs(output_dir, exist_ok=True)
    
    price_stats = df_prices.groupby(['store_id', 'item_id'])['sell_price'].agg(['max', 'min', 'mean']).reset_index()
    price_stats.columns = ['store_id', 'item_id', 'max_price', 'min_price', 'mean_price']
    
    df_prices_enhanced = df_prices.merge(price_stats, on=['store_id', 'item_id'])
    df_prices_enhanced['discount_pct'] = np.round(
        (df_prices_enhanced['max_price'] - df_prices_enhanced['sell_price']) / (df_prices_enhanced['max_price'] + 1e-5) * 100, 1
    )
    df_prices_enhanced['discount_tier'] = pd.cut(
        df_prices_enhanced['discount_pct'], 
        bins=[-1, 5, 15, 30, 100], 
        labels=['0-5% (Full Price)', '5-15% (Mild Markdown)', '15-30% (Moderate Promotion)', '30%+ (Deep Clearance)']
    )
    
    sample_items = np.random.choice(df_sales['id'].unique(), size=min(1000, len(df_sales)), replace=False)
    df_sales_sample = df_sales[df_sales['id'].isin(sample_items)].copy()
    
    date_cols = [c for c in df_sales.columns if c.startswith('d_')][-365:]
    melted = df_sales_sample.melt(id_vars=['id', 'item_id', 'cat_id', 'store_id'], value_vars=date_cols, var_name='d', value_name='sales')
    melted = melted.merge(df_calendar[['d', 'wm_yr_wk']], on='d')
    melted = melted.merge(df_prices_enhanced[['store_id', 'item_id', 'wm_yr_wk', 'sell_price', 'discount_tier']], on=['store_id', 'item_id', 'wm_yr_wk'])
    melted['revenue'] = melted['sales'] * melted['sell_price']
    
    elasticity = melted.groupby(['cat_id', 'discount_tier'], observed=True)[['sales', 'revenue']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(data=elasticity, x='cat_id', y='sales', hue='discount_tier', palette='viridis', ax=axes[0])
    axes[0].set_title("Constructed Feature: Mean Unit Sales Lift across Markdown Discount Tiers", fontweight='bold')
    axes[0].set_ylabel("Mean Daily Units Sold")
    axes[0].set_xlabel("Category")
    
    sns.barplot(data=elasticity, x='cat_id', y='revenue', hue='discount_tier', palette='magma', ax=axes[1])
    axes[1].set_title("Constructed Feature: Mean Gross Dollar Revenue ($) by Discount Tier", fontweight='bold')
    axes[1].set_ylabel("Mean Daily Revenue ($ USD)")
    axes[1].set_xlabel("Category")
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_4_composite_price_elasticity.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Composite Price Elasticity & Markdown plot saved to: {p1}")

def composite_calendar_payday_analysis(df_sales, df_calendar, output_dir="reports/figures"):
    """Constructs composite calendar features and analyzes cashflow timing surges."""
    print("\n[EDA] === Running 5. Composite Payday & Cashflow Cycle Analysis ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    total_sales_daily = df_sales[date_cols].sum().values
    
    df_cal_enhanced = df_calendar[df_calendar['d'].isin(date_cols)][['d', 'date', 'weekday', 'wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'event_name_1']].copy()
    df_cal_enhanced['date'] = pd.to_datetime(df_cal_enhanced['date'])
    df_cal_enhanced['day_of_month'] = df_cal_enhanced['date'].dt.day
    df_cal_enhanced['total_sales'] = total_sales_daily
    
    df_cal_enhanced['is_payday'] = df_cal_enhanced['day_of_month'].isin([1, 2, 15, 16, 30, 31]).map({True: 'Payday Window (1st/15th/Month-End)', False: 'Standard Mid-Month Days'})
    df_cal_enhanced['snap_concurrency'] = df_cal_enhanced['snap_CA'] + df_cal_enhanced['snap_TX'] + df_cal_enhanced['snap_WI']
    df_cal_enhanced['snap_status'] = df_cal_enhanced['snap_concurrency'].map({0: '0 States SNAP', 1: '1 State SNAP', 2: '2 States SNAP', 3: 'All 3 States SNAP Active'})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.boxplot(data=df_cal_enhanced, x='is_payday', y='total_sales', hue='is_payday', palette='Set2', ax=axes[0], legend=False)
    axes[0].set_title("Constructed Feature: Payday Cash-Flow Influx vs Total Sales Volume", fontweight='bold')
    axes[0].set_ylabel("Total Units Sold Across All Stores")
    axes[0].set_xlabel("")
    
    sns.barplot(data=df_cal_enhanced, x='snap_status', y='total_sales', hue='snap_status', palette='coolwarm', ax=axes[1], legend=False)
    axes[1].set_title("Constructed Feature: Multi-State SNAP Concurrency Demand Multiplier", fontweight='bold')
    axes[1].set_ylabel("Mean Units Sold Across All Stores")
    axes[1].set_xlabel("State SNAP Concurrency Level")
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_5_composite_payday_cashflow_cycles.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Composite Payday & Cashflow Cycle plot saved to: {p1}")

def composite_demand_classification_grid(df_sales, output_dir="reports/figures"):
    """Constructs Syntetos-Boylan Demand Categorization metrics (ADI vs CV^2)."""
    print("\n[EDA] === Running 6. Syntetos-Boylan Demand Categorization (ADI vs CV^2) ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')][-365:]
    sample_df = df_sales.sample(n=min(1500, len(df_sales)), random_state=42).copy()
    
    results = []
    for idx, row in sample_df.iterrows():
        series = row[date_cols].values.astype(float)
        non_zeros = series[series > 0]
        if len(non_zeros) < 2:
            continue
            
        adi = len(series) / len(non_zeros)
        cv2 = (np.std(non_zeros) / (np.mean(non_zeros) + 1e-5)) ** 2
        
        if adi < 1.32 and cv2 < 0.49:
            cat = "Smooth (Predictable)"
        elif adi >= 1.32 and cv2 < 0.49:
            cat = "Intermittent (Sporadic Count)"
        elif adi < 1.32 and cv2 >= 0.49:
            cat = "Erratic (Volatile Size)"
        else:
            cat = "Lumpy (Sporadic & Volatile)"
            
        results.append({
            'id': row['id'],
            'cat_id': row['cat_id'],
            'dept_id': row['dept_id'],
            'adi': adi,
            'cv2': cv2,
            'demand_type': cat
        })
        
    df_adi_cv2 = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    palette = {'Smooth (Predictable)': '#2ca02c', 'Intermittent (Sporadic Count)': '#1f77b4', 
               'Erratic (Volatile Size)': '#ff7f0e', 'Lumpy (Sporadic & Volatile)': '#d62728'}
    sns.scatterplot(data=df_adi_cv2, x='adi', y='cv2', hue='demand_type', palette=palette, alpha=0.6, ax=axes[0])
    axes[0].axvline(1.32, color='black', linestyle='--', alpha=0.7, label='ADI Cutoff (1.32)')
    axes[0].axhline(0.49, color='black', linestyle=':', alpha=0.7, label='CV² Cutoff (0.49)')
    axes[0].set_xlim(0.8, 6.0)
    axes[0].set_ylim(-0.1, 4.0)
    axes[0].set_title("Syntetos-Boylan Demand Profiling: ADI vs CV²", fontweight='bold')
    axes[0].set_xlabel("Average Demand Interval (ADI)")
    axes[0].set_ylabel("Squared Coefficient of Variation (CV²)")
    axes[0].legend(loc='upper right')
    
    breakdown = pd.crosstab(df_adi_cv2['cat_id'], df_adi_cv2['demand_type'], normalize='index') * 100
    breakdown.plot(kind='bar', stacked=True, ax=axes[1], color=['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c'])
    axes[1].set_title("Demand Categorization Profile by Product Category (%)", fontweight='bold')
    axes[1].set_ylabel("Percentage of Series (%)")
    axes[1].set_xlabel("Category")
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title="Demand Type", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_6_composite_demand_classification_grid.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Syntetos-Boylan Demand Categorization plot saved to: {p1}")

def composite_department_basket_mix(df_sales, df_calendar, output_dir="reports/figures"):
    """Constructs Department Basket Mix Share and analyzes seasonal wallet-share rotations."""
    print("\n[EDA] === Running 7. Department Basket Mix & Seasonal Share Analysis ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    
    dept_sales = df_sales.groupby('cat_id')[date_cols].sum().T.reset_index()
    dept_sales = dept_sales.merge(df_calendar[['d', 'month', 'year']], left_on='index', right_on='d')
    monthly_mix = dept_sales.groupby('month')[['FOODS', 'HOBBIES', 'HOUSEHOLD']].sum()
    monthly_share = monthly_mix.div(monthly_mix.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_share.plot(kind='area', stacked=True, alpha=0.8, colormap='Accent', ax=ax)
    ax.set_title("Constructed Feature: Monthly Product Category Wallet-Share Mix (% of Total Sales)", fontweight='bold')
    ax.set_ylabel("Share of Total Units Sold (%)")
    ax.set_xlabel("Month of Year (1=Jan ... 12=Dec)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylim(0, 100)
    ax.legend(title="Category", loc='lower right')
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_7_composite_dept_basket_mix.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Department Basket Mix Share plot saved to: {p1}")

# ==============================================================================
# MULTI-ATTRIBUTE GRID ANALYSES (DEEP-DIVE VARIANCE DECOMPOSITION)
# ==============================================================================

def grid_store_department_heterogeneity(df_sales, df_calendar, output_dir="reports/figures"):
    """
    Constructs a 10-Store multi-panel facet grid showing category sales trajectories
    across all individual store locations to analyze cross-store variance.
    """
    print("\n[EDA] === Running 8. Multi-Store & Category Heterogeneity Grid (10-Store Matrix) ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    
    stores = sorted(df_sales['store_id'].unique())
    fig, axes = plt.subplots(5, 2, figsize=(18, 16), sharex=True)
    axes = axes.flatten()
    
    date_map = df_calendar.set_index('d')['date'].to_dict()
    
    for i, store in enumerate(stores):
        ax = axes[i]
        store_data = df_sales[df_sales['store_id'] == store].groupby('cat_id')[date_cols].sum().T
        store_data.index = pd.to_datetime([date_map[d] for d in store_data.index])
        
        for cat in ['FOODS', 'HOUSEHOLD', 'HOBBIES']:
            if cat in store_data.columns:
                ax.plot(store_data.index, store_data[cat].rolling(30).mean(), label=cat, linewidth=1.8)
                
        state = store.split('_')[0]
        ax.set_title(f"Store: {store} ({state}) - 30-Day Moving Average by Category", fontweight='bold', fontsize=11)
        ax.set_ylabel("Daily Units")
        if i == 0:
            ax.legend(loc='upper left', fontsize=9)
            
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_8_store_department_heterogeneity_grid.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Multi-Store Heterogeneity Grid saved to: {p1}")

def grid_event_type_shock_spectrum(df_sales, df_calendar, output_dir="reports/figures"):
    """
    Constructs a 2x2 grid analyzing the shock response across event types:
    - Event Day lift vs non-event baseline
    - Holiday 14-day lead-up and post-event curve (-7 to +7 days)
    - State-level sensitivity to event categories
    - Top 10 individual holiday shock impact rankings
    """
    print("\n[EDA] === Running 9. Event Type Shock Spectrum & Lead-Up Grid (2x2 Matrix) ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    
    cat_daily = df_sales.groupby('cat_id')[date_cols].sum().T.reset_index()
    cat_daily.columns = ['d', 'FOODS', 'HOBBIES', 'HOUSEHOLD']
    merged_events = cat_daily.merge(df_calendar[['d', 'date', 'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']], on='d')
    merged_events['event_type_1'] = merged_events['event_type_1'].fillna('Regular Day')
    merged_events['event_name_1'] = merged_events['event_name_1'].fillna('None')
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Panel 1: Event Type Lift across Categories
    event_means = merged_events.groupby('event_type_1')[['FOODS', 'HOBBIES', 'HOUSEHOLD']].mean()
    baseline = event_means.loc['Regular Day']
    lift_pct = (event_means - baseline) / baseline * 100
    lift_pct = lift_pct.drop('Regular Day', errors='ignore')
    
    lift_pct.plot(kind='bar', ax=axes[0, 0], colormap='Set1')
    axes[0, 0].set_title("Panel 1: Event Type Sales Lift (%) vs. Regular Day Baseline", fontweight='bold')
    axes[0, 0].set_ylabel("% Sales Lift vs Baseline")
    axes[0, 0].set_xlabel("Event Type")
    axes[0, 0].tick_params(axis='x', rotation=15)
    axes[0, 0].axhline(0, color='black', linestyle='--')
    
    # Panel 2: Major Holiday Lead-Up Ramp Curve (-7 to +7 days around Thanksgiving)
    merged_events['date'] = pd.to_datetime(merged_events['date'])
    thanksgiving_days = merged_events[merged_events['event_name_1'] == 'Thanksgiving']['date']
    
    ramp_profiles = []
    for t_date in thanksgiving_days:
        window_start = t_date - pd.Timedelta(7, unit='D')
        window_end = t_date + pd.Timedelta(7, unit='D')
        sub = merged_events[(merged_events['date'] >= window_start) & (merged_events['date'] <= window_end)].copy()
        if len(sub) == 15:
            sub['relative_day'] = np.arange(-7, 8)
            ramp_profiles.append(sub[['relative_day', 'FOODS', 'HOUSEHOLD', 'HOBBIES']])
            
    if ramp_profiles:
        avg_ramp = pd.concat(ramp_profiles).groupby('relative_day').mean()
        axes[0, 1].plot(avg_ramp.index, avg_ramp['FOODS'], label='FOODS', marker='o', linewidth=2, color='#2ca02c')
        axes[0, 1].plot(avg_ramp.index, avg_ramp['HOUSEHOLD'], label='HOUSEHOLD', marker='s', linewidth=2, color='#1f77b4')
        axes[0, 1].plot(avg_ramp.index, avg_ramp['HOBBIES'], label='HOBBIES', marker='^', linewidth=2, color='#ff7f0e')
        axes[0, 1].axvline(0, color='red', linestyle='--', label='Thanksgiving Day')
        axes[0, 1].set_title("Panel 2: Pre-Holiday Stock-up Ramp vs Post-Holiday Drop (-7 to +7 Days)", fontweight='bold')
        axes[0, 1].set_xlabel("Days Relative to Holiday (Day 0 = Thanksgiving)")
        axes[0, 1].set_ylabel("Mean Units Sold")
        axes[0, 1].legend()
        
    # Panel 3: State-Level Reaction to Event Types
    state_daily = df_sales.groupby('state_id')[date_cols].sum().T.reset_index()
    state_daily.columns = ['d', 'CA', 'TX', 'WI']
    state_merged = state_daily.merge(df_calendar[['d', 'event_type_1']], on='d')
    state_merged['event_type_1'] = state_merged['event_type_1'].fillna('Regular Day')
    state_event_lift = state_merged.groupby('event_type_1')[['CA', 'TX', 'WI']].mean()
    state_baseline = state_event_lift.loc['Regular Day']
    state_lift = (state_event_lift - state_baseline) / state_baseline * 100
    state_lift = state_lift.drop('Regular Day', errors='ignore')
    
    state_lift.plot(kind='bar', ax=axes[1, 0], colormap='plasma')
    axes[1, 0].set_title("Panel 3: State-Specific Demand Sensitivity across Event Types (%)", fontweight='bold')
    axes[1, 0].set_ylabel("% Volume Variance vs Baseline")
    axes[1, 0].set_xlabel("Event Type")
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    # Panel 4: Top 10 Holiday Demand Spikes (Ranked by Total Volume Lift)
    event_specific = merged_events[merged_events['event_name_1'] != 'None'].groupby('event_name_1')[['FOODS', 'HOBBIES', 'HOUSEHOLD']].sum()
    event_specific['Total'] = event_specific.sum(axis=1)
    top10_events = event_specific.sort_values('Total', ascending=False).head(10)
    
    sns.barplot(data=top10_events.reset_index(), y='event_name_1', x='Total', palette='rocket', ax=axes[1, 1])
    axes[1, 1].set_title("Panel 4: Top 10 High-Impact Individual Events by Cumulative Volume", fontweight='bold')
    axes[1, 1].set_xlabel("Total Units Sold")
    axes[1, 1].set_ylabel("Event Name")
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_9_event_type_shock_spectrum_grid.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Event Type Shock Spectrum Grid saved to: {p1}")

def grid_lag_correlation_variance(df_sales, output_dir="reports/figures"):
    """
    Constructs a Multi-Lag Autocorrelation Grid across Stores and Departments:
    - Analyzes Pearson correlation with lags t-7, t-14, t-21, t-28, t-35, t-56, t-91, t-182, t-364.
    """
    print("\n[EDA] === Running 10. Multi-Lag Autocorrelation Matrix & Variance Spectrum ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    
    lags = [7, 14, 21, 28, 35, 56, 91, 182, 364]
    
    # Department Lag Correlations
    dept_ts = df_sales.groupby('cat_id')[date_cols].sum().T
    dept_corr_data = {}
    for cat in dept_ts.columns:
        s = dept_ts[cat]
        corrs = [s.corr(s.shift(l)) for l in lags]
        dept_corr_data[cat] = corrs
    df_dept_corr = pd.DataFrame(dept_corr_data, index=[f'Lag {l}d' for l in lags])
    
    # Store Lag Correlations
    store_ts = df_sales.groupby('store_id')[date_cols].sum().T
    store_corr_data = {}
    for store in sorted(store_ts.columns):
        s = store_ts[store]
        corrs = [s.corr(s.shift(l)) for l in lags]
        store_corr_data[store] = corrs
    df_store_corr = pd.DataFrame(store_corr_data, index=[f'Lag {l}d' for l in lags])
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Category Lag Correlation Heatmap
    sns.heatmap(df_dept_corr.T, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0.3, vmax=1.0, ax=axes[0])
    axes[0].set_title("Panel 1: Autocorrelation Memory Retention across Product Categories", fontweight='bold')
    axes[0].set_xlabel("Time Series Lag Distance")
    axes[0].set_ylabel("Category")
    
    # Store Lag Correlation Heatmap
    sns.heatmap(df_store_corr.T, annot=True, fmt='.2f', cmap='Blues', vmin=0.3, vmax=1.0, ax=axes[1])
    axes[1].set_title("Panel 2: Autocorrelation Memory Retention across All 10 Stores", fontweight='bold')
    axes[1].set_xlabel("Time Series Lag Distance")
    axes[1].set_ylabel("Store ID")
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_10_lag_correlation_variance_grid.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Multi-Lag Autocorrelation Grid saved to: {p1}")

def grid_zero_streaks_item_lifecycle(df_sales, df_calendar, df_prices, output_dir="reports/figures"):
    """
    Constructs a 3-panel grid analyzing:
    - Zero-sales streak lengths (consecutive days of shelf dwell without purchases)
    - Item Lifecycle Aging Curve (weeks since release into store)
    - Zero-streak vs Item Price dispersion
    """
    print("\n[EDA] === Running 11. Zero-Streak & Item Lifecycle Aging Grid ===")
    os.makedirs(output_dir, exist_ok=True)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')][-365:]
    
    sample_df = df_sales.sample(n=min(1000, len(df_sales)), random_state=42).copy()
    
    streak_records = []
    for idx, row in sample_df.iterrows():
        series = row[date_cols].values.astype(int)
        zeros = (series == 0).astype(int)
        from itertools import groupby
        zero_runs = [len(list(g)) for k, g in groupby(zeros) if k == 1]
        max_streak = max(zero_runs) if zero_runs else 0
        mean_streak = np.mean(zero_runs) if zero_runs else 0
        streak_records.append({
            'id': row['id'],
            'cat_id': row['cat_id'],
            'store_id': row['store_id'],
            'max_zero_streak': max_streak,
            'mean_zero_streak': mean_streak
        })
    df_streaks = pd.DataFrame(streak_records)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Distribution of Mean Zero Streak Lengths by Category
    sns.boxplot(data=df_streaks, x='cat_id', y='mean_zero_streak', hue='cat_id', palette='Set2', ax=axes[0], legend=False)
    axes[0].set_title("Panel 1: Average Consecutive Days with 0 Sales (Shelf Dwell Time)", fontweight='bold')
    axes[0].set_ylabel("Mean Zero-Streak Length (Days)")
    axes[0].set_xlabel("Category")
    
    # Panel 2: Max Zero Streak Distribution (Survival / Tail Risk)
    for cat in ['FOODS', 'HOUSEHOLD', 'HOBBIES']:
        sub = df_streaks[df_streaks['cat_id'] == cat]['max_zero_streak']
        sns.kdeplot(sub, ax=axes[1], label=cat, fill=True, alpha=0.3)
    axes[1].set_title("Panel 2: Max Zero-Streak Tail Distribution (Stockout / Stagnation Risk)", fontweight='bold')
    axes[1].set_xlabel("Maximum Consecutive Zero Days in Year")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    # Panel 3: Item Release Velocity (First Sale to Maturity Lifecycle)
    first_sale_days = []
    for idx, row in sample_df.head(300).iterrows():
        arr = row[[c for c in df_sales.columns if c.startswith('d_')]].values
        non_zero_indices = np.where(arr > 0)[0]
        if len(non_zero_indices) > 0:
            first_idx = non_zero_indices[0]
            after_launch = arr[first_idx: first_idx + 60]
            if len(after_launch) == 60:
                first_sale_days.append(after_launch)
                
    if first_sale_days:
        launch_mat = np.array(first_sale_days)
        mean_ramp = np.mean(launch_mat, axis=0)
        axes[2].plot(range(1, 61), mean_ramp, color='#1f77b4', linewidth=2)
        axes[2].plot(range(1, 61), pd.Series(mean_ramp).rolling(7).mean(), color='#d62728', linewidth=2.5, label='7-Day Smooth')
        axes[2].set_title("Panel 3: New Item Post-Launch Adoption Velocity (First 60 Days)", fontweight='bold')
        axes[2].set_xlabel("Days Since First Recorded Store Sale")
        axes[2].set_ylabel("Mean Units Sold")
        axes[2].legend()
        
    plt.tight_layout()
    p1 = os.path.join(output_dir, "eda_11_zero_streaks_item_lifecycle_grid.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"   [SAVED] Zero-Streak & Lifecycle Aging Grid saved to: {p1}")

def run_full_eda(raw_data_dir="data/raw", output_dir="reports/figures"):
    """Orchestrates the complete 11-part Exploratory, Composite & Multi-Attribute Grid EDA Suite."""
    print("[EDA SUITE] Initializing Complete 11-Part Exploratory & Multi-Attribute Grid Analysis...")
    
    sales_path = os.path.join(raw_data_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(raw_data_dir, "calendar.csv")
    prices_path = os.path.join(raw_data_dir, "sell_prices.csv")
    
    df_sales = pd.read_csv(sales_path)
    df_calendar = pd.read_csv(calendar_path)
    df_prices = pd.read_csv(prices_path)
    
    # 1-3: Standard Foundations
    univariate_analysis(df_sales, df_prices, output_dir=output_dir)
    bivariate_analysis(df_sales, df_calendar, df_prices, output_dir=output_dir)
    multivariate_time_series_analysis(df_sales, df_calendar, output_dir=output_dir)
    
    # 4-7: Advanced Composite Feature Trend Analyses
    composite_price_elasticity_analysis(df_sales, df_calendar, df_prices, output_dir=output_dir)
    composite_calendar_payday_analysis(df_sales, df_calendar, output_dir=output_dir)
    composite_demand_classification_grid(df_sales, output_dir=output_dir)
    composite_department_basket_mix(df_sales, df_calendar, output_dir=output_dir)
    
    # 8-11: Multi-Attribute Variance & Cross-Column Grids
    grid_store_department_heterogeneity(df_sales, df_calendar, output_dir=output_dir)
    grid_event_type_shock_spectrum(df_sales, df_calendar, output_dir=output_dir)
    grid_lag_correlation_variance(df_sales, output_dir=output_dir)
    grid_zero_streaks_item_lifecycle(df_sales, df_calendar, df_prices, output_dir=output_dir)
    
    print("\n[SUCCESS] Complete 11-Part EDA Suite finished! All 11 visual diagnostic grids saved to:", os.path.abspath(output_dir))

if __name__ == "__main__":
    run_full_eda()
