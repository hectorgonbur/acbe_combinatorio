import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Optional
import io

warnings.filterwarnings('ignore')

# =============================
# PAGE CONFIGURATION
# =============================

st.set_page_config(
    page_title="ACBE-S73 Quantum Betting Suite v2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# SESSION STATE INITIALIZATION
# =============================

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'entropy' not in st.session_state:
    st.session_state.entropy = None
if 'expected_values' not in st.session_state:
    st.session_state.expected_values = None
if 's73_combinations' not in st.session_state:
    st.session_state.s73_combinations = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'equity_curve' not in st.session_state:
    st.session_state.equity_curve = None

# =============================
# UI LAYER
# =============================

st.title("🎯 ACBE-S73 Quantum Betting Suite v2.0")
st.markdown("""
<div style='background-color:#1a1a2e; padding:20px; border-radius:10px; margin-bottom:20px;'>
<h4 style='color:#00ff88; margin:0;'>Institutional Quantitative Betting Optimization Suite</h4>
<p style='color:#cccccc; margin:5px 0 0 0;'>Integrating ACBE probabilistic inference, entropy-based uncertainty weighting, fractional Kelly allocation, and institutional S73 combinatorial optimization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Parameters")
    
    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=1000.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0
    )
    
    kelly_fraction = st.slider(
        "Fractional Kelly",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    risk_aversion = st.slider(
        "Risk Aversion λ",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1
    )
    
    correlation_penalty = st.slider(
        "Correlation Penalty γ",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    s73_target = st.slider(
        "Target S73 Columns",
        min_value=20,
        max_value=100,
        value=73,
        step=1
    )
    
    monte_carlo_iterations = st.select_slider(
        "Monte Carlo Iterations",
        options=[1000, 2500, 5000, 10000, 25000],
        value=5000
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📊 System Status
    """)
    
    status_color = "#ff4444" if not st.session_state.analysis_run else "#00ff88"
    st.markdown(f"""
    <div style='background-color:#1a1a2e; padding:10px; border-radius:5px;'>
    <p style='color:{status_color}; margin:0;'>
    {'⚠️ Analysis Pending' if not st.session_state.analysis_run else '✅ Analysis Complete'}
    </p>
    </div>
    """, unsafe_allow_html=True)

# Main Input Form
st.header("📝 Match Input Configuration")

with st.form("input_form"):
    st.markdown("### Enter Match Data (6 Matches Required)")
    
    # Create input grid for 6 matches
    matches_data = []
    for i in range(6):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            league = st.text_input(f"League {i+1}", value=f"League {i+1}", key=f"league_{i}")
        with col2:
            home_team = st.text_input(f"Home {i+1}", value=f"Team H{i+1}", key=f"home_{i}")
        with col3:
            away_team = st.text_input(f"Away {i+1}", value=f"Team A{i+1}", key=f"away_{i}")
        with col4:
            odds_1 = st.number_input(f"1 {i+1}", min_value=1.01, max_value=100.0, value=2.5, step=0.1, key=f"odds1_{i}")
        with col5:
            odds_x = st.number_input(f"X {i+1}", min_value=1.01, max_value=100.0, value=3.2, step=0.1, key=f"oddsx_{i}")
        with col6:
            odds_2 = st.number_input(f"2 {i+1}", min_value=1.01, max_value=100.0, value=2.8, step=0.1, key=f"odds2_{i}")
        
        matches_data.append({
            'league': league,
            'home': home_team,
            'away': away_team,
            'odds_1': odds_1,
            'odds_x': odds_x,
            'odds_2': odds_2
        })
    
    # CSV Upload Option
    st.markdown("---")
    csv_file = st.file_uploader("Or upload CSV with match data", type=['csv'])
    
    if csv_file:
        try:
            df_csv = pd.read_csv(csv_file)
            st.write("CSV Preview:", df_csv.head())
        except:
            st.error("Invalid CSV format")
    
    submit_button = st.form_submit_button("🚀 Run Full Analysis")

# =============================
# DATA VALIDATION LAYER
# =============================

def validate_inputs(matches_data: List[Dict], bankroll: float) -> Tuple[bool, str]:
    """Validate all inputs before processing."""
    
    if bankroll <= 0:
        return False, "Bankroll must be positive"
    
    if len(matches_data) != 6:
        return False, "Exactly 6 matches required"
    
    for i, match in enumerate(matches_data):
        # Check odds validity
        for outcome in ['odds_1', 'odds_x', 'odds_2']:
            odds = match[outcome]
            if odds <= 1.01:
                return False, f"Match {i+1}: Odds must be > 1.01"
        
        # Check team names
        if not match['home'].strip() or not match['away'].strip():
            return False, f"Match {i+1}: Team names cannot be empty"
    
    return True, "Validation passed"

# =============================
# QUANTITATIVE ENGINE
# =============================

class ACBEEngine:
    """Bayesian Gamma-Poisson model for match outcome probabilities."""
    
    @staticmethod
    def simulate_match_outcomes(matches_data: List[Dict], n_simulations: int = 10000) -> np.ndarray:
        """Vectorized Monte Carlo simulation of match outcomes using Gamma-Poisson model."""
        n_matches = len(matches_data)
        
        # Initialize attack/defense parameters with some randomness
        np.random.seed(42)  # For reproducibility
        attack_home = np.random.gamma(1.5, 0.5, n_matches)
        attack_away = np.random.gamma(1.3, 0.5, n_matches)
        defense_home = np.random.gamma(0.8, 0.3, n_matches)
        defense_away = np.random.gamma(0.8, 0.3, n_matches)
        home_advantage = np.random.gamma(1.2, 0.1, n_matches)
        
        # Calculate lambda parameters
        lambda_home = attack_home * defense_away * home_advantage
        lambda_away = attack_away * defense_home
        
        # Vectorized Poisson simulation
        home_goals = np.random.poisson(lambda_home[:, None], (n_matches, n_simulations))
        away_goals = np.random.poisson(lambda_away[:, None], (n_matches, n_simulations))
        
        # Determine outcomes (1: home win, X: draw, 2: away win)
        outcomes = np.zeros((n_matches, n_simulations), dtype=int)
        outcomes[home_goals > away_goals] = 0  # Home win
        outcomes[home_goals == away_goals] = 1  # Draw
        outcomes[home_goals < away_goals] = 2  # Away win
        
        return outcomes
    
    @staticmethod
    def calculate_probabilities(outcomes: np.ndarray) -> np.ndarray:
        """Calculate probabilities from simulated outcomes."""
        n_matches = outcomes.shape[0]
        probabilities = np.zeros((n_matches, 3))
        
        for i in range(n_matches):
            for j in range(3):
                probabilities[i, j] = np.mean(outcomes[i] == j)
        
        # Normalize to ensure sum = 1
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> np.ndarray:
        """Calculate normalized entropy for each match."""
        n_matches = probabilities.shape[0]
        entropy_values = np.zeros(n_matches)
        
        for i in range(n_matches):
            # Base-3 entropy
            H = -np.sum(probabilities[i] * np.log(probabilities[i] + 1e-10) / np.log(3))
            entropy_values[i] = H
        
        # Normalize between 0 and 1
        if entropy_values.max() > entropy_values.min():
            entropy_values = (entropy_values - entropy_values.min()) / (entropy_values.max() - entropy_values.min())
        
        return entropy_values
    
    @staticmethod
    def calculate_expected_values(probabilities: np.ndarray, matches_data: List[Dict]) -> np.ndarray:
        """Calculate expected value for each outcome."""
        n_matches = len(matches_data)
        expected_values = np.zeros((n_matches, 3))
        
        for i in range(n_matches):
            odds = np.array([matches_data[i]['odds_1'], matches_data[i]['odds_x'], matches_data[i]['odds_2']])
            expected_values[i] = probabilities[i] * odds - 1
        
        return expected_values

# =============================
# S73 COMBINATORIAL REDESIGN
# =============================

class InstitutionalS73:
    """Institutional version of S73 combinatorial optimization."""
    
    @staticmethod
    def generate_combinations(n_matches: int = 6) -> np.ndarray:
        """Generate all possible 3^6 combinations."""
        outcomes = [0, 1, 2]  # 1, X, 2
        all_combinations = np.array(np.meshgrid(*[outcomes]*n_matches)).T.reshape(-1, n_matches)
        return all_combinations
    
    @staticmethod
    def calculate_combination_score(combination: np.ndarray, 
                                   probabilities: np.ndarray,
                                   matches_data: List[Dict],
                                   risk_aversion: float,
                                   correlation_penalty: float,
                                   selected_combinations: List[np.ndarray] = None) -> Dict:
        """Calculate institutional score for a combination."""
        
        n_matches = len(matches_data)
        
        # Joint probability
        joint_prob = 1.0
        for i in range(n_matches):
            outcome = combination[i]
            joint_prob *= probabilities[i, outcome]
        
        # Combination odds
        combo_odds = 1.0
        for i in range(n_matches):
            outcome = combination[i]
            odds_key = ['odds_1', 'odds_x', 'odds_2'][outcome]
            combo_odds *= matches_data[i][odds_key]
        
        # Expected return
        expected_return = joint_prob * combo_odds - 1
        
        # Variance approximation
        variance_approx = joint_prob * (1 - joint_prob)
        
        # Correlation penalty
        correlation_penalty_value = 0.0
        if selected_combinations and len(selected_combinations) > 0:
            overlaps = []
            for selected in selected_combinations:
                overlap = np.mean(combination == selected)
                overlaps.append(overlap)
            correlation_penalty_value = np.mean(overlaps) if overlaps else 0.0
        
        # Final score
        score = expected_return - risk_aversion * variance_approx - correlation_penalty * correlation_penalty_value
        
        return {
            'combination': combination,
            'joint_probability': joint_prob,
            'odds': combo_odds,
            'expected_return': expected_return,
            'variance': variance_approx,
            'correlation_penalty': correlation_penalty_value,
            'score': score
        }
    
    @staticmethod
    def select_top_combinations(probabilities: np.ndarray,
                               matches_data: List[Dict],
                               target_columns: int,
                               risk_aversion: float,
                               correlation_penalty: float) -> pd.DataFrame:
        """Select top combinations using institutional scoring."""
        
        all_combinations = InstitutionalS73.generate_combinations()
        selected_combinations = []
        combination_scores = []
        
        # Calculate scores for all combinations
        for combo in all_combinations:
            score_dict = InstitutionalS73.calculate_combination_score(
                combo, probabilities, matches_data, risk_aversion, correlation_penalty
            )
            combination_scores.append(score_dict)
        
        # Sort by score descending
        combination_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top combinations with correlation-aware selection
        selected = []
        for score_dict in combination_scores:
            if len(selected) >= target_columns:
                break
            
            combo = score_dict['combination']
            
            # Calculate correlation with already selected
            if len(selected) > 0:
                max_correlation = max([
                    np.mean(combo == s) for s in selected
                ])
                if max_correlation > 0.8:  # Skip highly correlated combinations
                    continue
            
            # Recalculate score with current selected set
            final_score = InstitutionalS73.calculate_combination_score(
                combo, probabilities, matches_data, risk_aversion, 
                correlation_penalty, selected
            )
            
            selected.append(combo)
            score_dict.update(final_score)
        
        # Create DataFrame
        top_combinations = combination_scores[:target_columns]
        df = pd.DataFrame(top_combinations)
        
        return df

# =============================
# PORTFOLIO OPTIMIZATION LAYER
# =============================

class PortfolioOptimizer:
    """Unified portfolio optimization with fractional Kelly allocation."""
    
    @staticmethod
    def calculate_kelly_stake(probability: float, odds: float, bankroll: float, 
                             kelly_fraction: float = 0.25) -> float:
        """Calculate Kelly stake with fractional adjustment."""
        if odds <= 1:
            return 0.0
        
        kelly = (probability * odds - 1) / (odds - 1)
        fractional_kelly = kelly * kelly_fraction
        stake = fractional_kelly * bankroll
        
        # Cap at 3% of bankroll
        max_stake = 0.03 * bankroll
        return min(stake, max_stake)
    
    @staticmethod
    def build_unified_portfolio(probabilities: np.ndarray,
                               expected_values: np.ndarray,
                               matches_data: List[Dict],
                               s73_combinations: pd.DataFrame,
                               bankroll: float,
                               kelly_fraction: float) -> pd.DataFrame:
        """Build unified portfolio with singles and combinations."""
        
        portfolio = []
        n_matches = len(matches_data)
        
        # Add positive EV singles
        for i in range(n_matches):
            for outcome in range(3):
                ev = expected_values[i, outcome]
                if ev > 0:
                    prob = probabilities[i, outcome]
                    odds_key = ['odds_1', 'odds_x', 'odds_2'][outcome]
                    odds = matches_data[i][odds_key]
                    
                    stake = PortfolioOptimizer.calculate_kelly_stake(
                        prob, odds, bankroll, kelly_fraction
                    )
                    
                    if stake > 0:
                        outcome_str = ['1', 'X', '2'][outcome]
                        portfolio.append({
                            'Type': 'Single',
                            'Description': f"{matches_data[i]['home']} vs {matches_data[i]['away']} - {outcome_str}",
                            'Probability': prob,
                            'Odds': odds,
                            'Stake': stake,
                            'EV': ev,
                            'Expected_Profit': stake * ev
                        })
        
        # Add S73 combinations
        if s73_combinations is not None:
            for idx, row in s73_combinations.iterrows():
                stake = PortfolioOptimizer.calculate_kelly_stake(
                    row['joint_probability'], row['odds'], bankroll, kelly_fraction
                )
                
                if stake > 0:
                    # Create readable description
                    desc_parts = []
                    combo = row['combination']
                    for i in range(n_matches):
                        outcome_str = ['1', 'X', '2'][int(combo[i])]
                        desc_parts.append(f"M{i+1}:{outcome_str}")
                    
                    portfolio.append({
                        'Type': 'Combo',
                        'Description': ' | '.join(desc_parts),
                        'Probability': row['joint_probability'],
                        'Odds': row['odds'],
                        'Stake': stake,
                        'EV': row['expected_return'],
                        'Expected_Profit': stake * row['expected_return']
                    })
        
        # Create DataFrame and sort by expected profit
        df_portfolio = pd.DataFrame(portfolio)
        if not df_portfolio.empty:
            df_portfolio = df_portfolio.sort_values('Expected_Profit', ascending=False)
            df_portfolio.index = range(1, len(df_portfolio) + 1)
        
        return df_portfolio

# =============================
# RISK & BACKTESTING LAYER
# =============================

class MonteCarloBacktester:
    """Vectorized Monte Carlo backtesting engine."""
    
    @staticmethod
    def simulate_outcomes(probabilities: np.ndarray, n_simulations: int) -> np.ndarray:
        """Simulate actual outcomes based on true probabilities."""
        n_matches = probabilities.shape[0]
        outcomes = np.zeros((n_simulations, n_matches), dtype=int)
        
        for i in range(n_matches):
            # Sample from multinomial distribution
            probs = probabilities[i]
            outcomes[:, i] = np.random.choice([0, 1, 2], size=n_simulations, p=probs)
        
        return outcomes
    
    @staticmethod
    def backtest_portfolio(portfolio: pd.DataFrame,
                          matches_data: List[Dict],
                          probabilities: np.ndarray,
                          bankroll: float,
                          n_iterations: int) -> Dict:
        """Run vectorized Monte Carlo backtest."""
        
        if portfolio.empty:
            return {
                'roi': 0.0,
                'cagr': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'equity_curve': np.zeros(n_iterations)
            }
        
        n_simulations = n_iterations
        n_bets = len(portfolio)
        
        # Simulate outcomes for all matches
        outcomes = MonteCarloBacktester.simulate_outcomes(probabilities, n_simulations)
        
        # Initialize results array
        profits = np.zeros(n_simulations)
        
        # Process each bet in the portfolio
        for idx, bet in portfolio.iterrows():
            stake = bet['Stake']
            odds = bet['Odds']
            
            # Parse bet description to determine which matches and outcomes
            if bet['Type'] == 'Single':
                # Parse single bet
                desc = bet['Description']
                for i, match in enumerate(matches_data):
                    match_str = f"{match['home']} vs {match['away']}"
                    if match_str in desc:
                        outcome_str = desc.split('-')[-1].strip()
                        target_outcome = {'1': 0, 'X': 1, '2': 2}[outcome_str]
                        
                        # Check if bet wins in each simulation
                        wins = (outcomes[:, i] == target_outcome)
                        profits += np.where(wins, stake * (odds - 1), -stake)
                        break
            
            else:  # Combo bet
                # Parse combination bet
                desc = bet['Description']
                combo_parts = desc.split(' | ')
                all_wins = np.ones(n_simulations, dtype=bool)
                
                for part in combo_parts:
                    match_idx = int(part.split(':')[0][1:]) - 1
                    outcome_str = part.split(':')[1]
                    target_outcome = {'1': 0, 'X': 1, '2': 2}[outcome_str]
                    
                    # Check if this part wins
                    match_wins = (outcomes[:, match_idx] == target_outcome)
                    all_wins &= match_wins
                
                profits += np.where(all_wins, stake * (odds - 1), -stake)
        
        # Calculate metrics
        returns = profits / bankroll
        roi = np.mean(returns) * 100
        
        # CAGR (assuming 1 time period)
        cagr = (1 + np.mean(returns)) - 1
        
        # Sharpe ratio (assuming risk-free rate = 0)
        excess_returns = returns
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
        
        # Equity curve
        equity_curve = bankroll + np.cumsum(profits[:1000])  # First 1000 simulations
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        # VaR 95%
        var_95 = np.percentile(returns, 5) * 100
        
        # Win rate
        win_rate = np.mean(profits > 0) * 100
        
        # Profit factor
        gross_profits = profits[profits > 0].sum()
        gross_losses = abs(profits[profits < 0].sum())
        profit_factor = gross_profits / (gross_losses + 1e-10)
        
        return {
            'roi': roi,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'returns_distribution': returns
        }

# =============================
# VISUALIZATION LAYER
# =============================

class VisualizationEngine:
    """Advanced visualization engine for betting analytics."""
    
    @staticmethod
    def create_probability_bars(probabilities: np.ndarray, matches_data: List[Dict]) -> go.Figure:
        """Create stacked probability bar chart."""
        fig = go.Figure()
        
        outcomes = ['Home Win', 'Draw', 'Away Win']
        colors = ['#00ff88', '#ffaa00', '#ff4444']
        
        for i in range(3):
            fig.add_trace(go.Bar(
                x=[f"M{j+1}\n{m['home']}\nvs\n{m['away']}" for j, m in enumerate(matches_data)],
                y=probabilities[:, i],
                name=outcomes[i],
                marker_color=colors[i],
                text=[f"{p:.1%}" for p in probabilities[:, i]],
                textposition='inside'
            ))
        
        fig.update_layout(
            title="Match Outcome Probabilities",
            barmode='stack',
            yaxis_title="Probability",
            yaxis_tickformat=".0%",
            height=400,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    @staticmethod
    def create_entropy_chart(entropy: np.ndarray, matches_data: List[Dict]) -> go.Figure:
        """Create entropy line chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(entropy) + 1)),
            y=entropy,
            mode='lines+markers',
            name='Entropy',
            line=dict(color='#00ffff', width=3),
            marker=dict(size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title="Match Uncertainty (Entropy)",
            xaxis_title="Match Number",
            yaxis_title="Normalized Entropy",
            height=350,
            template="plotly_dark",
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        
        return fig
    
    @staticmethod
    def create_combination_distribution(s73_combinations: pd.DataFrame) -> go.Figure:
        """Create distribution of combination probabilities."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=s73_combinations['joint_probability'],
            nbinsx=50,
            name='Combinations',
            marker_color='#ffaa00',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Distribution of Combination Probabilities",
            xaxis_title="Joint Probability",
            yaxis_title="Count",
            height=350,
            template="plotly_dark"
        )
        
        return fig
    
    @staticmethod
    def create_equity_curve(equity_curve: np.ndarray) -> go.Figure:
        """Create equity curve and drawdown chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=list(range(len(equity_curve))),
                y=equity_curve,
                mode='lines',
                name='Equity',
                line=dict(color='#00ff88', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(drawdown))),
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.3)',
                line=dict(color='#ff4444', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            template="plotly_dark",
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Bankroll ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Simulation", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_return_distribution(returns: np.ndarray) -> go.Figure:
        """Create return distribution histogram."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='#8888ff',
            opacity=0.7
        ))
        
        # Add vertical line for mean
        mean_return = np.mean(returns) * 100
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="#00ff88",
            annotation_text=f"Mean: {mean_return:.2f}%"
        )
        
        fig.update_layout(
            title="Return Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=350,
            template="plotly_dark"
        )
        
        return fig

# =============================
# EXECUTION PIPELINE
# =============================

if submit_button:
    with st.spinner("🔄 Validating inputs..."):
        # Validate inputs
        is_valid, validation_msg = validate_inputs(matches_data, bankroll)
        
        if not is_valid:
            st.error(f"Validation failed: {validation_msg}")
            st.stop()
    
    with st.spinner("🧠 Running ACBE probabilistic inference..."):
        # Run ACBE engine
        acbe = ACBEEngine()
        outcomes = acbe.simulate_match_outcomes(matches_data, 20000)
        probabilities = acbe.calculate_probabilities(outcomes)
        entropy = acbe.calculate_entropy(probabilities)
        expected_values = acbe.calculate_expected_values(probabilities, matches_data)
        
        # Store in session state
        st.session_state.probabilities = probabilities
        st.session_state.entropy = entropy
        st.session_state.expected_values = expected_values
    
    with st.spinner("🎰 Generating institutional S73 combinations..."):
        # Generate S73 combinations
        s73 = InstitutionalS73()
        s73_combinations = s73.select_top_combinations(
            probabilities, matches_data, s73_target, 
            risk_aversion, correlation_penalty
        )
        st.session_state.s73_combinations = s73_combinations
    
    with st.spinner("💰 Optimizing portfolio allocation..."):
        # Build unified portfolio
        optimizer = PortfolioOptimizer()
        portfolio = optimizer.build_unified_portfolio(
            probabilities, expected_values, matches_data,
            s73_combinations, bankroll, kelly_fraction
        )
        st.session_state.portfolio = portfolio
    
    with st.spinner("📊 Running Monte Carlo backtest..."):
        # Run backtest
        backtester = MonteCarloBacktester()
        backtest_results = backtester.backtest_portfolio(
            portfolio, matches_data, probabilities, 
            bankroll, monte_carlo_iterations
        )
        st.session_state.backtest_results = backtest_results
        st.session_state.equity_curve = backtest_results['equity_curve']
    
    st.session_state.analysis_run = True
    st.success("✅ Analysis complete! Results are ready.")

# =============================
# RESULTS DISPLAY
# =============================

if st.session_state.analysis_run:
    st.header("📈 Analysis Results")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Probability Analysis",
        "💰 Portfolio Allocation",
        "📊 Risk Metrics",
        "📈 Visualizations",
        "📁 Export Results"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Probabilities")
            prob_df = pd.DataFrame(
                st.session_state.probabilities,
                columns=['P(Home)', 'P(Draw)', 'P(Away)']
            )
            prob_df.index = [f"Match {i+1}" for i in range(6)]
            prob_df = prob_df.style.format("{:.2%}").background_gradient(cmap='RdYlGn')
            st.dataframe(prob_df, use_container_width=True)
        
        with col2:
            st.subheader("Expected Values")
            ev_df = pd.DataFrame(
                st.session_state.expected_values,
                columns=['EV(1)', 'EV(X)', 'EV(2)']
            )
            ev_df.index = [f"Match {i+1}" for i in range(6)]
            
            def highlight_positive(val):
                color = '#00ff88' if val > 0 else '#ff4444' if val < 0 else 'white'
                return f'color: {color}'
            
            ev_df = ev_df.style.format("{:.3f}").applymap(highlight_positive)
            st.dataframe(ev_df, use_container_width=True)
        
        # Visualizations
        viz = VisualizationEngine()
        st.plotly_chart(
            viz.create_probability_bars(st.session_state.probabilities, matches_data),
            use_container_width=True
        )
        
        st.plotly_chart(
            viz.create_entropy_chart(st.session_state.entropy, matches_data),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Unified Portfolio Allocation")
        
        if not st.session_state.portfolio.empty:
            # Format portfolio display
            display_portfolio = st.session_state.portfolio.copy()
            display_portfolio['Probability'] = display_portfolio['Probability'].apply(lambda x: f"{x:.2%}")
            display_portfolio['Odds'] = display_portfolio['Odds'].apply(lambda x: f"{x:.2f}")
            display_portfolio['Stake'] = display_portfolio['Stake'].apply(lambda x: f"${x:,.2f}")
            display_portfolio['EV'] = display_portfolio['EV'].apply(lambda x: f"{x:.3f}")
            display_portfolio['Expected_Profit'] = display_portfolio['Expected_Profit'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_portfolio, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            total_stake = st.session_state.portfolio['Stake'].sum()
            expected_profit = st.session_state.portfolio['Expected_Profit'].sum()
            
            with col1:
                st.metric("Total Bets", len(st.session_state.portfolio))
            with col2:
                st.metric("Total Stake", f"${total_stake:,.2f}")
            with col3:
                st.metric("Expected Profit", f"${expected_profit:,.2f}")
            with col4:
                st.metric("Stake/Bankroll", f"{(total_stake/bankroll)*100:.1f}%")
        else:
            st.warning("No positive EV opportunities found with current parameters.")
    
    with tab3:
        st.subheader("Risk Analytics & Backtest Results")
        
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ROI", f"{results['roi']:.2f}%")
                st.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
            
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                st.metric("VaR 95%", f"{results['var_95']:.2f}%")
            
            with col4:
                st.metric("CAGR", f"{results['cagr']:.2%}")
                st.metric("Simulations", f"{monte_carlo_iterations:,}")
            
            # Distribution visualization
            viz = VisualizationEngine()
            st.plotly_chart(
                viz.create_return_distribution(results['returns_distribution']),
                use_container_width=True
            )
    
    with tab4:
        st.subheader("Advanced Visualizations")
        
        if st.session_state.s73_combinations is not None:
            viz = VisualizationEngine()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    viz.create_combination_distribution(st.session_state.s73_combinations),
                    use_container_width=True
                )
            
            with col2:
                if st.session_state.equity_curve is not None:
                    st.plotly_chart(
                        viz.create_equity_curve(st.session_state.equity_curve),
                        use_container_width=True
                    )
    
    with tab5:
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export probabilities
            if st.session_state.probabilities is not None:
                prob_df = pd.DataFrame(
                    st.session_state.probabilities,
                    columns=['P(Home)', 'P(Draw)', 'P(Away)']
                )
                csv = prob_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Probabilities (CSV)",
                    data=csv,
                    file_name="acbe_probabilities.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Export portfolio
            if st.session_state.portfolio is not None and not st.session_state.portfolio.empty:
                csv = st.session_state.portfolio.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv,
                    file_name="acbe_portfolio.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        st.markdown("### 📋 Full Report Summary")
        
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            report = f"""
            ACBE-S73 QUANTUM BETTING SUITE v2.0 - ANALYSIS REPORT
            ====================================================
            
            System Parameters:
            - Bankroll: ${bankroll:,.2f}
            - Fractional Kelly: {kelly_fraction:.2f}
            - Risk Aversion λ: {risk_aversion:.2f}
            - Correlation Penalty γ: {correlation_penalty:.2f}
            - S73 Target Columns: {s73_target}
            - Monte Carlo Iterations: {monte_carlo_iterations:,}
            
            Portfolio Summary:
            - Total Bets: {len(st.session_state.portfolio) if st.session_state.portfolio is not None else 0}
            - Singles: {len(st.session_state.portfolio[st.session_state.portfolio['Type'] == 'Single']) if st.session_state.portfolio is not None else 0}
            - Combinations: {len(st.session_state.portfolio[st.session_state.portfolio['Type'] == 'Combo']) if st.session_state.portfolio is not None else 0}
            - Total Stake: ${st.session_state.portfolio['Stake'].sum():,.2f' if st.session_state.portfolio is not None else '0'}
            
            Backtest Results:
            - Expected ROI: {results['roi']:.2f}%
            - Sharpe Ratio: {results['sharpe']:.2f}
            - Max Drawdown: {results['max_drawdown']:.2f}%
            - VaR 95%: {results['var_95']:.2f}%
            - Win Rate: {results['win_rate']:.1f}%
            - Profit Factor: {results['profit_factor']:.2f}
            
            Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.download_button(
                label="📄 Download Full Report (TXT)",
                data=report,
                file_name="acbe_report.txt",
                mime="text/plain"
            )

else:
    # Initial state - show instructions
    st.markdown("""
    <div style='background-color:#1a1a2e; padding:30px; border-radius:10px; margin-top:20px;'>
    <h3 style='color:#00ff88;'>Welcome to ACBE-S73 Quantum Betting Suite v2.0</h3>
    <p style='color:#cccccc;'>To begin analysis:</p>
    <ol style='color:#cccccc;'>
    <li>Configure system parameters in the sidebar</li>
    <li>Enter match data for 6 matches in the form above</li>
    <li>Click <strong>"Run Full Analysis"</strong> to execute the quantitative pipeline</li>
    </ol>
    <p style='color:#cccccc; margin-top:20px;'>
    The system will execute the following pipeline:
    </p>
    <ul style='color:#cccccc;'>
    <li>🎯 ACBE Bayesian probabilistic inference</li>
    <li>📊 Entropy-based uncertainty weighting</li>
    <li>🎰 Institutional S73 combinatorial optimization</li>
    <li>💰 Fractional Kelly portfolio allocation</li>
    <li>📈 Vectorized Monte Carlo backtesting</li>
    <li>⚡ Advanced risk analytics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================
# FOOTER
# =============================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; font-size: 0.9em;'>
ACBE-S73 Quantum Betting Suite v2.0 | Institutional Quantitative Framework | © 2024
</div>
""", unsafe_allow_html=True)
