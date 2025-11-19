"""
Portfolio Risk Analysis - Complete EDA Code
This script performs comprehensive exploratory data analysis on a credit portfolio dataset

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(file_path):
    """
    Load and prepare the portfolio data from Excel file
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing portfolio data
    
    Returns:
    --------
    dict : Dictionary containing processed dataframes
    """
    
    # Load data skipping the header row
    df = pd.read_excel(file_path, skiprows=1)
    
    # Extract different data components
    data_dict = {
        'borrower_ids': df.iloc[:, 0],
        'numeric_years': df.iloc[:, 1:11],  # Numeric values for years 1-10
        'rating_years': df.iloc[:, 11:21],  # Rating values for years 1-10
        'amounts': pd.to_numeric(df.iloc[:, 21], errors='coerce'),
        'lgd': pd.to_numeric(df.iloc[:, 31], errors='coerce'),
        'ead_multiplier': pd.to_numeric(df.iloc[:, 41], errors='coerce')
    }
    
    # Set proper column names
    data_dict['numeric_years'].columns = [f'Year_{i}' for i in range(1, 11)]
    data_dict['rating_years'].columns = [f'Rating_Year_{i}' for i in range(1, 11)]
    
    return data_dict

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_rating_distribution(rating_years, rating_order):
    """
    Calculate rating distribution across years
    
    Parameters:
    -----------
    rating_years : DataFrame
        DataFrame containing ratings for each year
    rating_order : list
        Ordered list of rating categories
    
    Returns:
    --------
    DataFrame : Rating counts and percentages by year
    """
    
    rating_counts_by_year = pd.DataFrame()
    
    for i in range(1, 11):
        col = f'Rating_Year_{i}'
        counts = rating_years[col].value_counts()
        rating_counts_by_year[f'Year {i}'] = counts
    
    rating_counts_by_year = rating_counts_by_year.reindex(rating_order).fillna(0)
    rating_percentages = rating_counts_by_year.div(rating_counts_by_year.sum(axis=0), axis=1) * 100
    
    return rating_counts_by_year, rating_percentages

def calculate_migration_matrix(rating_years, rating_order):
    """
    Calculate rating migration probability matrix
    
    Parameters:
    -----------
    rating_years : DataFrame
        DataFrame containing ratings for each year
    rating_order : list
        Ordered list of rating categories
    
    Returns:
    --------
    ndarray : Migration probability matrix
    """
    
    migration_matrix = np.zeros((8, 8))
    
    for idx in range(len(rating_years)):
        for year in range(1, 10):
            current_rating = rating_years.iloc[idx, year-1]
            next_rating = rating_years.iloc[idx, year]
            
            if pd.notna(current_rating) and pd.notna(next_rating):
                try:
                    curr_idx = rating_order.index(current_rating)
                    next_idx = rating_order.index(next_rating)
                    migration_matrix[curr_idx, next_idx] += 1
                except ValueError:
                    continue
    
    # Convert to probability matrix
    migration_prob = migration_matrix / migration_matrix.sum(axis=1, keepdims=True)
    migration_prob = np.nan_to_num(migration_prob)
    
    return migration_prob

def calculate_risk_scores(rating_years, risk_scores_map):
    """
    Calculate portfolio risk scores over time
    
    Parameters:
    -----------
    rating_years : DataFrame
        DataFrame containing ratings for each year
    risk_scores_map : dict
        Mapping of ratings to risk scores
    
    Returns:
    --------
    tuple : Lists of average risk, 25th percentile, and 75th percentile
    """
    
    avg_risk_by_year = []
    percentile_25 = []
    percentile_75 = []
    
    for i in range(1, 11):
        col = f'Rating_Year_{i}'
        scores = rating_years[col].map(risk_scores_map).dropna()
        avg_risk_by_year.append(scores.mean())
        percentile_25.append(scores.quantile(0.25))
        percentile_75.append(scores.quantile(0.75))
    
    return avg_risk_by_year, percentile_25, percentile_75

def analyze_concentration(amounts):
    """
    Analyze exposure concentration in the portfolio
    
    Parameters:
    -----------
    amounts : Series
        Series containing exposure amounts
    
    Returns:
    --------
    Series : Cumulative exposure percentages
    """
    
    valid_amounts = amounts[amounts.notna() & (amounts > 0)]
    
    if len(valid_amounts) > 0:
        sorted_amounts = valid_amounts.sort_values(ascending=False)
        cumulative = sorted_amounts.cumsum() / sorted_amounts.sum() * 100
        return cumulative
    
    return pd.Series()

def calculate_migration_dynamics(rating_years, risk_scores):
    """
    Calculate upgrade/downgrade/stable percentages over time
    
    Parameters:
    -----------
    rating_years : DataFrame
        DataFrame containing ratings for each year
    risk_scores : dict
        Mapping of ratings to risk scores
    
    Returns:
    --------
    tuple : Lists of upgrade, downgrade, and stable percentages
    """
    
    upgrades_by_year = []
    downgrades_by_year = []
    stable_by_year = []
    
    for year in range(1, 10):
        upgrades = 0
        downgrades = 0
        stable = 0
        
        for idx in range(len(rating_years)):
            current = rating_years.iloc[idx, year-1]
            next_year = rating_years.iloc[idx, year]
            
            if pd.notna(current) and pd.notna(next_year):
                if current in risk_scores and next_year in risk_scores:
                    curr_score = risk_scores[current]
                    next_score = risk_scores[next_year]
                    
                    if next_score < curr_score:
                        upgrades += 1
                    elif next_score > curr_score:
                        downgrades += 1
                    else:
                        stable += 1
        
        total = upgrades + downgrades + stable
        if total > 0:
            upgrades_by_year.append(upgrades/total * 100)
            downgrades_by_year.append(downgrades/total * 100)
            stable_by_year.append(stable/total * 100)
    
    return upgrades_by_year, downgrades_by_year, stable_by_year

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comprehensive_dashboard(data_dict, output_path):
    """
    Create comprehensive dashboard with all visualizations
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed data
    output_path : str
        Path to save the dashboard image
    """
    
    # Set visualization style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Define constants
    rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'C', 'D']
    risk_scores = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4, 'BB': 5, 'B': 6, 'C': 7, 'D': 8}
    investment_grade = ['AAA', 'AA', 'A', 'BBB']
    
    # Extract data
    rating_years = data_dict['rating_years']
    amounts = data_dict['amounts']
    lgd = data_dict['lgd']
    ead_multiplier = data_dict['ead_multiplier']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # VISUALIZATION 1: Rating Evolution Heatmap
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    _, rating_percentages = calculate_rating_distribution(rating_years, rating_order)
    
    im = ax1.imshow(rating_percentages.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=30)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels([f'Y{i}' for i in range(1, 11)])
    ax1.set_yticks(range(8))
    ax1.set_yticklabels(rating_order)
    ax1.set_title('Rating Distribution Evolution (% of Portfolio)', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    for i in range(8):
        for j in range(10):
            text = ax1.text(j, i, f'{rating_percentages.iloc[i, j]:.0f}%', 
                           ha="center", va="center", color="black", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Percentage', rotation=270, labelpad=15)
    
    # ========================================================================
    # VISUALIZATION 2: Rating Migration Matrix
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    migration_prob = calculate_migration_matrix(rating_years, rating_order)
    
    im2 = ax2.imshow(migration_prob, cmap='YlOrRd', aspect='equal', vmin=0, vmax=0.5)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels([r[:3] for r in rating_order], rotation=45)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([r[:3] for r in rating_order])
    ax2.set_title('Rating Migration Matrix', fontsize=12, fontweight='bold')
    ax2.set_xlabel('To Rating', fontsize=10)
    ax2.set_ylabel('From Rating', fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # ========================================================================
    # VISUALIZATION 3: Portfolio Risk Score Evolution
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    avg_risk, p25, p75 = calculate_risk_scores(rating_years, risk_scores)
    
    years = range(1, 11)
    ax3.fill_between(years, p25, p75, alpha=0.3, color='coral', label='25th-75th Percentile')
    ax3.plot(years, avg_risk, 'o-', color='darkred', linewidth=2.5, markersize=8, label='Average Risk Score')
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Risk Score (1=AAA, 8=D)', fontsize=11)
    ax3.set_title('Portfolio Risk Score Trajectory', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xticks(years)
    
    # ========================================================================
    # VISUALIZATION 4: Exposure Concentration Analysis
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    cumulative = analyze_concentration(amounts)
    
    if len(cumulative) > 0:
        ax4.fill_between(range(len(cumulative)), cumulative.values, alpha=0.4, color='skyblue')
        ax4.plot(cumulative.values, color='navy', linewidth=2)
        
        # Mark concentration points
        pct_20 = int(len(cumulative) * 0.2)
        pct_50 = int(len(cumulative) * 0.5)
        ax4.axvline(x=pct_20, color='red', linestyle='--', alpha=0.7, 
                   label=f'Top 20%: {cumulative.iloc[pct_20]:.1f}%')
        ax4.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    
    ax4.set_xlabel('Number of Borrowers', fontsize=11)
    ax4.set_ylabel('Cumulative Exposure (%)', fontsize=11)
    ax4.set_title('Exposure Concentration Curve', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # VISUALIZATION 5: Risk-Adjusted Exposure Distribution
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    latest_ratings = rating_years.iloc[:, -1]
    
    rating_exposure = pd.DataFrame({
        'Rating': latest_ratings,
        'Amount': amounts,
        'LGD': lgd,
        'EAD_Mult': ead_multiplier
    }).dropna()
    
    if len(rating_exposure) > 0:
        rating_exposure['Risk_Adjusted_Exposure'] = (
            rating_exposure['Amount'] * 
            rating_exposure['LGD'] * 
            rating_exposure['EAD_Mult'] / 100
        )
        
        grouped = rating_exposure.groupby('Rating')['Risk_Adjusted_Exposure'].sum()
        grouped = grouped.reindex(rating_order, fill_value=0)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grouped)))
        bars = ax5.bar(range(len(grouped)), grouped.values, color=colors, 
                      edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(grouped)))
        ax5.set_xticklabels(grouped.index, rotation=45)
        ax5.set_ylabel('Risk-Adjusted Exposure (₹ Cr)', fontsize=11)
        ax5.set_title('Risk-Adjusted Exposure by Rating', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, grouped.values):
            if val > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # ========================================================================
    # VISUALIZATION 6: Portfolio Quality Trend
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 0])
    ig_percentages = []
    speculative_percentages = []
    default_percentages = []
    
    for i in range(1, 11):
        col = f'Rating_Year_{i}'
        total = rating_years[col].notna().sum()
        ig = rating_years[col].isin(investment_grade).sum()
        default = (rating_years[col] == 'D').sum()
        spec = total - ig - default
        
        ig_percentages.append(ig/total * 100)
        speculative_percentages.append(spec/total * 100)
        default_percentages.append(default/total * 100)
    
    width = 0.65
    x_pos = np.arange(10)
    p1 = ax6.bar(x_pos, ig_percentages, width, label='Investment Grade', 
                color='green', alpha=0.8)
    p2 = ax6.bar(x_pos, speculative_percentages, width, bottom=ig_percentages, 
                label='Speculative', color='orange', alpha=0.8)
    p3 = ax6.bar(x_pos, default_percentages, width, 
                bottom=[i+j for i,j in zip(ig_percentages, speculative_percentages)], 
                label='Default', color='red', alpha=0.8)
    
    ax6.set_xlabel('Year', fontsize=11)
    ax6.set_ylabel('Portfolio Composition (%)', fontsize=11)
    ax6.set_title('Portfolio Quality Evolution', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'Y{i}' for i in range(1, 11)])
    ax6.legend(loc='upper right')
    ax6.set_ylim(0, 100)
    
    # ========================================================================
    # VISUALIZATION 7: Rating Volatility Analysis
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 1])
    volatility_scores = []
    
    for idx in range(len(rating_years)):
        row_ratings = rating_years.iloc[idx].map(risk_scores).dropna()
        if len(row_ratings) > 1:
            volatility_scores.append(row_ratings.std())
    
    volatility_scores = pd.Series(volatility_scores)
    
    ax7.hist(volatility_scores, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax7.axvline(volatility_scores.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {volatility_scores.mean():.2f}')
    ax7.axvline(volatility_scores.median(), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {volatility_scores.median():.2f}')
    ax7.set_xlabel('Rating Volatility (Std Dev)', fontsize=11)
    ax7.set_ylabel('Number of Borrowers', fontsize=11)
    ax7.set_title('Distribution of Rating Volatility', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # ========================================================================
    # VISUALIZATION 8: Downgrade/Upgrade Balance
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    upgrades, downgrades, stable = calculate_migration_dynamics(rating_years, risk_scores)
    
    years = range(1, len(upgrades) + 1)
    ax8.plot(years, upgrades, 'g-', marker='o', linewidth=2, label='Upgrades')
    ax8.plot(years, downgrades, 'r-', marker='s', linewidth=2, label='Downgrades')
    ax8.plot(years, stable, 'b-', marker='^', linewidth=2, label='Stable')
    ax8.fill_between(years, 0, upgrades, alpha=0.3, color='green')
    ax8.fill_between(years, 0, downgrades, alpha=0.3, color='red')
    ax8.set_xlabel('Year Transition', fontsize=11)
    ax8.set_ylabel('Percentage of Portfolio (%)', fontsize=11)
    ax8.set_title('Rating Migration Dynamics', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks(years)
    ax8.set_xticklabels([f'{i}→{i+1}' for i in years])
    
    # Final touches
    plt.suptitle('Portfolio Risk Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def generate_summary_statistics(data_dict):
    """
    Generate comprehensive summary statistics for the portfolio
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed data
    
    Returns:
    --------
    dict : Dictionary containing summary statistics
    """
    
    rating_years = data_dict['rating_years']
    amounts = data_dict['amounts']
    lgd = data_dict['lgd']
    ead_multiplier = data_dict['ead_multiplier']
    
    # Define constants
    rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'C', 'D']
    risk_scores = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4, 'BB': 5, 'B': 6, 'C': 7, 'D': 8}
    investment_grade = ['AAA', 'AA', 'A', 'BBB']
    
    stats = {}
    
    # Portfolio size statistics
    stats['total_borrowers'] = len(rating_years)
    stats['total_exposure'] = amounts.sum()
    stats['avg_exposure'] = amounts.mean()
    stats['median_exposure'] = amounts.median()
    
    # Rating distribution statistics (latest year)
    latest_ratings = rating_years.iloc[:, -1]
    rating_dist = latest_ratings.value_counts()
    stats['rating_distribution'] = rating_dist.to_dict()
    
    # Investment grade vs speculative grade
    ig_count = latest_ratings.isin(investment_grade).sum()
    total_count = latest_ratings.notna().sum()
    stats['investment_grade_pct'] = (ig_count / total_count) * 100
    stats['speculative_grade_pct'] = ((total_count - ig_count) / total_count) * 100
    
    # Risk metrics
    latest_risk_scores = latest_ratings.map(risk_scores).dropna()
    stats['avg_risk_score'] = latest_risk_scores.mean()
    stats['risk_score_std'] = latest_risk_scores.std()
    
    # Concentration metrics
    valid_amounts = amounts[amounts.notna() & (amounts > 0)]
    if len(valid_amounts) > 0:
        sorted_amounts = valid_amounts.sort_values(ascending=False)
        cumulative = sorted_amounts.cumsum() / sorted_amounts.sum()
        
        # Herfindahl-Hirschman Index (HHI)
        market_shares = valid_amounts / valid_amounts.sum()
        stats['hhi'] = (market_shares ** 2).sum() * 10000
        
        # Concentration ratios
        stats['cr_top10'] = cumulative.iloc[min(9, len(cumulative)-1)] * 100
        stats['cr_top20'] = cumulative.iloc[min(19, len(cumulative)-1)] * 100
        stats['cr_top50'] = cumulative.iloc[min(49, len(cumulative)-1)] * 100
    
    # LGD and EAD statistics
    stats['avg_lgd'] = lgd.mean()
    stats['avg_ead_multiplier'] = ead_multiplier.mean()
    
    # Migration statistics
    upgrades_total = 0
    downgrades_total = 0
    stable_total = 0
    
    for idx in range(len(rating_years)):
        for year in range(1, 10):
            current = rating_years.iloc[idx, year-1]
            next_year = rating_years.iloc[idx, year]
            
            if pd.notna(current) and pd.notna(next_year):
                if current in risk_scores and next_year in risk_scores:
                    curr_score = risk_scores[current]
                    next_score = risk_scores[next_year]
                    
                    if next_score < curr_score:
                        upgrades_total += 1
                    elif next_score > curr_score:
                        downgrades_total += 1
                    else:
                        stable_total += 1
    
    total_migrations = upgrades_total + downgrades_total + stable_total
    if total_migrations > 0:
        stats['upgrade_rate'] = (upgrades_total / total_migrations) * 100
        stats['downgrade_rate'] = (downgrades_total / total_migrations) * 100
        stats['stability_rate'] = (stable_total / total_migrations) * 100
    
    return stats

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function for portfolio EDA
    """
    
    # File paths
    input_file = '/mnt/user-data/uploads/framassingment.xlsx'
    output_dashboard = '/mnt/user-data/outputs/portfolio_eda_dashboard.png'
    output_stats = '/mnt/user-data/outputs/portfolio_summary_statistics.txt'
    
    print("Starting Portfolio Risk Analysis...")
    print("-" * 50)
    
    # Load and prepare data
    print("Loading data...")
    data_dict = load_and_prepare_data(input_file)
    print(f"Loaded {len(data_dict['rating_years'])} borrowers across 10 years")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    fig = create_comprehensive_dashboard(data_dict, output_dashboard)
    print(f"Dashboard saved to: {output_dashboard}")
    
    # Generate summary statistics
    print("\nCalculating summary statistics...")
    stats = generate_summary_statistics(data_dict)
    
    # Save statistics to file
    with open(output_stats, 'w') as f:
        f.write("PORTFOLIO SUMMARY STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Portfolio Size Metrics:\n")
        f.write(f"  Total Borrowers: {stats['total_borrowers']:,}\n")
        f.write(f"  Total Exposure: ₹{stats['total_exposure']:,.2f} Cr\n")
        f.write(f"  Average Exposure: ₹{stats['avg_exposure']:,.2f} Cr\n")
        f.write(f"  Median Exposure: ₹{stats['median_exposure']:,.2f} Cr\n\n")
        
        f.write("Credit Quality Metrics:\n")
        f.write(f"  Investment Grade: {stats['investment_grade_pct']:.1f}%\n")
        f.write(f"  Speculative Grade: {stats['speculative_grade_pct']:.1f}%\n")
        f.write(f"  Average Risk Score: {stats['avg_risk_score']:.2f}\n")
        f.write(f"  Risk Score Std Dev: {stats['risk_score_std']:.2f}\n\n")
        
        f.write("Concentration Metrics:\n")
        f.write(f"  HHI Index: {stats.get('hhi', 0):.1f}\n")
        f.write(f"  Top 10 Concentration: {stats.get('cr_top10', 0):.1f}%\n")
        f.write(f"  Top 20 Concentration: {stats.get('cr_top20', 0):.1f}%\n")
        f.write(f"  Top 50 Concentration: {stats.get('cr_top50', 0):.1f}%\n\n")
        
        f.write("Migration Dynamics:\n")
        f.write(f"  Upgrade Rate: {stats.get('upgrade_rate', 0):.1f}%\n")
        f.write(f"  Downgrade Rate: {stats.get('downgrade_rate', 0):.1f}%\n")
        f.write(f"  Stability Rate: {stats.get('stability_rate', 0):.1f}%\n\n")
        
        f.write("Risk Parameters:\n")
        f.write(f"  Average LGD: {stats['avg_lgd']:.2f}%\n")
        f.write(f"  Average EAD Multiplier: {stats['avg_ead_multiplier']:.2f}x\n")
    
    print(f"Statistics saved to: {output_stats}")
    
    # Print summary to console
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"\nKey Findings:")
    print(f"  • Portfolio contains {stats['total_borrowers']:,} borrowers")
    print(f"  • Total exposure: ₹{stats['total_exposure']:,.0f} Crore")
    print(f"  • Investment Grade: {stats['investment_grade_pct']:.1f}%")
    print(f"  • Downgrade Rate: {stats.get('downgrade_rate', 0):.1f}%")
    print(f"  • Top 20 borrowers control {stats.get('cr_top20', 0):.1f}% of exposure")
    print("\nAll outputs saved successfully!")

# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    main()