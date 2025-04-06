import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from portfolio_monte_carlo import simulate_portfolio_monte_carlo
from llm_api import chat_with_gpt4o, chat_with_deepseek
from portfolio_analysis import get_ai_recommended_actions
import json
import time
import random
import yfinance as yf
from datetime import datetime, timedelta

# Helper function to format currency values in Indian format (K for thousands, L for lakhs)
def format_currency(value):
    """
    Format currency values in Indian format:
    - If value is in thousands (≥1,000 and <100,000), show as xK
    - If value is in lakhs (≥100,000), show as x.xxL
    
    Args:
        value (float): The monetary value to format
        
    Returns:
        str: Formatted value with K or L suffix
    """
    abs_value = abs(value)
    if abs_value >= 100000:  # 1 lakh or more
        return f"₹{abs_value/100000:.2f}L"
    elif abs_value >= 1000:  # 1 thousand or more
        return f"₹{abs_value/1000:.0f}K"
    else:
        return f"₹{abs_value:.2f}"

# Set page configuration
st.set_page_config(
    page_title="Portfolio Monte Carlo Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, animated design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
        animation: gradientBG 15s ease infinite;
        background-size: 400% 400%;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Card styling with animations - Dark Theme */
    .card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        cursor: pointer;
        border-left: 4px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0) 30%, rgba(255,255,255,0.1), rgba(255,255,255,0) 70%);
        transform: translateX(-100%);
        transition: 0.6s;
    }
    
    .card:hover::before {
        transform: translateX(100%);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #4299E1;
    }
    
    .card-container {
        position: relative;
        cursor: pointer;
    }
    
    .card-selected {
        border-left: 4px solid #4CAF50;
        background-color: #263850;
        box-shadow: 0 8px 16px rgba(76, 175, 80, 0.3);
    }
    
    .card-title {
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 1.1em;
        color: #E2E8F0;
    }
    
    .card-description {
        color: #A0AEC0;
        font-size: 0.9em;
        line-height: 1.4;
    }
    
    /* Metric cards with animations - Dark Theme */
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #63B3ED;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover .metric-value {
        transform: scale(1.05);
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #A0AEC0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .positive {
        color: #38A169;
    }
    
    .negative {
        color: #E53E3E;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        margin-bottom: 30px;
        font-weight: 700;
        color: #1a202c;
        font-size: 2.5em;
        background: linear-gradient(120deg, #2B6CB0, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .subheader {
        font-size: 1.2em;
        color: #4a5568;
        margin-bottom: 30px;
        text-align: center;
        font-weight: 400;
        animation: fadeIn 1s ease-in 0.2s both;
    }
    
    /* Section headers */
    h2 {
        font-weight: 600;
        color: #2D3748;
        margin-top: 40px;
        margin-bottom: 20px;
        position: relative;
        display: inline-block;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 40px;
        height: 3px;
        background: linear-gradient(90deg, #2B6CB0, #4CAF50);
        transition: width 0.3s ease;
    }
    
    h2:hover::after {
        width: 100%;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    
    /* Event and sector badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.7em;
        font-weight: 600;
        margin-right: 5px;
        color: white;
    }
    
    .badge-tech {
        background-color: #4299E1;
    }
    
    .badge-finance {
        background-color: #805AD5;
    }
    
    .badge-energy {
        background-color: #F6AD55;
    }
    
    .badge-high {
        background-color: #38A169;
    }
    
    .badge-medium {
        background-color: #ECC94B;
    }
    
    .badge-low {
        background-color: #F56565;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 60px;
        padding: 20px;
        color: #718096;
        font-size: 0.9em;
        border-top: 1px solid #E2E8F0;
    }
    
    /* Animations for content sections */
    @keyframes slideInFromLeft {
        0% { transform: translateX(-30px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInFromRight {
        0% { transform: translateX(30px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInFromBottom {
        0% { transform: translateY(30px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    .animate-left {
        animation: slideInFromLeft 0.5s ease-out forwards;
    }
    
    .animate-right {
        animation: slideInFromRight 0.5s ease-out forwards;
    }
    
    .animate-bottom {
        animation: slideInFromBottom 0.5s ease-out forwards;
    }
    
    /* Streamlit branding - footer hidden but menu visible */
    /* #MainMenu {visibility: hidden;} */
    footer {visibility: hidden;}
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #2D3748;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Hide buttons in card containers, but not the custom scenario button */
    .card-container .stButton > button {
        display: none !important;
        opacity: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }
    
    /* Ensure the Generate Custom Scenario button is visible */
    button[kind="primary"], 
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button#generate_custom_scenario_btn,
    div[data-testid="column"]:nth-of-type(2) .stButton > button,
    .stButton > button[key="generate_custom_scenario_btn"] {
        display: inline-flex !important;
        opacity: 1 !important;
        visibility: visible !important;
        height: auto !important;
        padding: 0.25rem 0.75rem !important;
        margin: 0.25rem !important;
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
        background-color: #4299E1 !important;
        color: white !important;
        z-index: 999 !important;
        position: relative !important;
    }
    
    /* Make the card container take full width */
    .card-container .stButton {
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: 1;
        margin: 0 !important;
        padding: 0 !important;
    }
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Add animation classes to elements
    const addAnimationClasses = () => {
        const elements = document.querySelectorAll('.metric-card');
        elements.forEach((el, index) => {
            el.classList.add('animate-bottom');
            el.style.animationDelay = `${index * 0.1}s`;
        });
    };
    
    // Call the function
    addAnimationClasses();
    
    // Add event listeners for cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('click', function() {
            this.classList.toggle('card-selected');
        });
    });
});
</script>
""", unsafe_allow_html=True)

# Define a sample portfolio
@st.cache_data
def get_default_portfolio():
    return {
        "RELIANCE.NS": {
            "quantity": 10,
            "buy_price": 2500,
            "sector": "Energy"
        },
        "TCS.NS": {
            "quantity": 5,
            "buy_price": 3200,
            "sector": "Technology"
        },
        "HDFCBANK.NS": {
            "quantity": 20,
            "buy_price": 1600,
            "sector": "Financial"
        },
        "INFY.NS": {
            "quantity": 15,
            "buy_price": 1400,
            "sector": "Technology"
        },
        "SBIN.NS": {
            "quantity": 30,
            "buy_price": 550,
            "sector": "Financial"
        }
    }

# Define events with descriptions and icons
@st.cache_data
def get_events():
    return [
        {
            "id": "market_correction",
            "type": "market_correction",
            "title": "Market Correction",
            "description": "A 10-15% market correction due to overvaluation concerns",
            "icon": "📉",
            "probability": "High",
            "month": 1  # Occurs in 2nd month
        },
        {
            "id": "us_tariff_increase",
            "type": "us_tariff_increase",
            "title": "Geopolitical Trade Tensions",
            "description": "US increases tariffs on imports affecting global trade",
            "icon": "🌐",
            "probability": "High",
            "month": 2  # Occurs in 1st month
        },
        {
            "id": "rbi_rate_cut",
            "type": "rbi_rate_cut",
            "title": "RBI Rate Cut",
            "description": "RBI cuts interest rates by 25-50 basis points to stimulate growth",
            "icon": "🏦",
            "probability": "Medium",
            "month": 3  # Occurs in 4th month
        },
        {
            "id": "bull_run",
            "type": "bull_run",
            "title": "Year-End Bull Run",
            "description": "Strong market rally in the last quarter of the year",
            "icon": "🐂",
            "probability": "Medium",
            "month": 10  # Occurs in 10th month
        },
        {
            "id": "energy_sector_boom",
            "type": "energy_sector_boom",
            "title": "Increase in FIIs Limits",
            "description": "Regulatory reforms to double the investment limits for FIIs",
            "icon": "⚡",
            "probability": "High",
            "month": 6  # Occurs in 6th month
        },
        
        {
            "id": "rbi_rate_hike",
            "type": "rbi_rate_hike",
            "title": "Global Economic Slowdown",
            "description": "Concerns over a potential global recession",
            "icon": "🐢",
            "probability": "Medium",
            "month": 5  # Occurs in 5th month
        }
    ]

# Define actions with descriptions and icons
@st.cache_data
def get_actions():
    return [
        {
            "id": "add_lumpsum_all",
            "type": "add_lumpsum_all",
            "title": "Add ₹100,000 Lumpsum",
            "description": "Invest an additional ₹100,000 across all stocks proportionally",
            "icon": "💰",
            "amount": 100000,
            "month": 3  # Occurs in 3rd month
        },
        {
            "id": "add_lumpsum_tcs",
            "type": "add_lumpsum_stock",
            "title": "Add ₹50,000 to TCS",
            "description": "Invest an additional ₹50,000 in TCS shares",
            "icon": "🖥️",
            "ticker": "TCS.NS",
            "amount": 50000,
            "month": 7  # Occurs in 7th month
        },
        {
            "id": "book_profit_reliance",
            "type": "book_profit",
            "title": "Book 30% Profit in Reliance",
            "description": "Sell 30% of your Reliance holdings to book profit",
            "icon": "📊",
            "ticker": "RELIANCE.NS",
            "percentage": 30,
            "month": 9  # Occurs in 9th month
        }
    ]

# Run simulation without events or actions
@st.cache_data
def run_base_simulation(portfolio_data):
    results = simulate_portfolio_monte_carlo(
        portfolio_data=portfolio_data,
        events=[],
        actions=[],
        forecast_days=252,  # 1 year
        num_simulations=500,
        seed=42,  # For reproducibility
        plot_results=False
    )
    return results

# Run simulation with selected events and actions
# Remove caching to ensure fresh data each time
def run_simulation_with_selections(portfolio_data, selected_events, selected_actions):
    # Convert selected events and actions to the format expected by the simulation
    events = []
    for event_id in selected_events:
        # First check in predefined events
        event_found = False
        for event in get_events():
            if event["id"] == event_id:
                # Convert month to trading days (approximately 21 trading days per month)
                day = (event["month"] - 1) * 21 + 10  # Middle of the month
                events.append({
                    "type": event["type"],
                    "description": event["description"],
                    "day": day  # Specify the exact day for the event
                })
                event_found = True
                break
        
        # If not found in predefined events, check in custom scenarios
        if not event_found and 'custom_scenarios' in st.session_state:
            for custom_item in st.session_state.custom_scenarios:
                scenario = custom_item["scenario"]
                if scenario["id"] == event_id and custom_item["type"] == "event":
                    # Convert month to trading days
                    day = (scenario["month"] - 1) * 21 + 10
                    events.append({
                        "type": scenario["type"],
                        "description": scenario["description"],
                        "day": day
                    })
                    break
    
    actions = []
    for action_id in selected_actions:
        # First check in predefined actions
        action_found = False
        for action in get_actions():
            if action["id"] == action_id:
                # Convert month to trading days (approximately 21 trading days per month)
                day = (action["month"] - 1) * 21 + 10  # Middle of the month
                action_data = {
                    "type": action["type"],
                    "day": day  # Specify the exact day for the action
                }
                if "ticker" in action:
                    action_data["ticker"] = action["ticker"]
                if "amount" in action:
                    action_data["amount"] = action["amount"]
                if "percentage" in action:
                    action_data["percentage"] = action["percentage"]
                actions.append(action_data)
                action_found = True
                break
        
        # If not found in predefined actions, check in custom scenarios and AI recommendations
        if not action_found:
            # Check in custom scenarios
            if 'custom_scenarios' in st.session_state:
                for custom_item in st.session_state.custom_scenarios:
                    scenario = custom_item["scenario"]
                    if scenario["id"] == action_id and custom_item["type"] == "action":
                        # Convert month to trading days
                        day = (scenario["month"] - 1) * 21 + 10
                        action_data = {
                            "type": scenario["type"],
                            "day": day
                        }
                        if "ticker" in scenario:
                            action_data["ticker"] = scenario["ticker"]
                        if "amount" in scenario:
                            action_data["amount"] = scenario["amount"]
                        if "percentage" in scenario:
                            action_data["percentage"] = scenario["percentage"]
                        actions.append(action_data)
                        action_found = True
                        break
            
            # Check in AI recommendations
            if not action_found and 'ai_recommendations' in st.session_state:
                for ai_action in st.session_state.ai_recommendations:
                    if ai_action["id"] == action_id:
                        # Convert month to trading days
                        day = (ai_action["month"] - 1) * 21 + 10
                        action_data = {
                            "type": ai_action["type"],
                            "day": day
                        }
                        if "ticker" in ai_action:
                            action_data["ticker"] = ai_action["ticker"]
                        if "amount" in ai_action:
                            action_data["amount"] = ai_action["amount"]
                        if "percentage" in ai_action:
                            action_data["percentage"] = ai_action["percentage"]
                        actions.append(action_data)
                        break
    
    results = simulate_portfolio_monte_carlo(
        portfolio_data=portfolio_data,
        events=events,
        actions=actions,
        forecast_days=252,  # 1 year
        num_simulations=500,
        seed=42,  # For reproducibility
        plot_results=False
    )
    return results

# Function to fetch historical data and calculate historical portfolio returns
@st.cache_data
def get_historical_portfolio_returns(portfolio_data):
    """
    Fetch historical price data for the portfolio stocks from 6 months ago and calculate returns.
    
    Args:
        portfolio_data (dict): The portfolio data with stock details
        
    Returns:
        tuple: (historical_days, historical_returns, historical_dates)
    """
    try:
        # Calculate the date 6 months ago from today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # Approximately 6 months
        
        # Fetch historical data for each stock in the portfolio
        historical_prices = {}
        for ticker, details in portfolio_data.items():
            try:
                # Fetch historical data
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock_data.empty:
                    historical_prices[ticker] = stock_data['Close']
            except Exception as e:
                print(f"Error fetching historical data for {ticker}: {e}")
        
        # Check if we have any data
        if not historical_prices:
            print("Warning: Could not fetch historical data for any stocks")
            return None, None, None
        
        # Create a DataFrame with the historical prices
        # Make sure we have a proper index by using the index from the first series
        first_ticker = list(historical_prices.keys())[0]
        common_index = historical_prices[first_ticker].index
        
        # Create DataFrame with the common index
        price_df = pd.DataFrame(index=common_index)
        
        # Add each ticker's price series to the DataFrame
        for ticker, prices in historical_prices.items():
            price_df[ticker] = prices
        
        # Resample to business days and forward fill missing values
        price_df = price_df.resample('B').ffill()
        
        # Calculate portfolio value over time
        portfolio_values = []
        for date, prices in price_df.iterrows():
            day_value = 0
            for ticker, price in prices.items():
                if not pd.isna(price) and ticker in portfolio_data:
                    day_value += portfolio_data[ticker]['quantity'] * price
            portfolio_values.append(day_value)
        
        # Check if we have any portfolio values
        if not portfolio_values:
            print("Warning: Could not calculate portfolio values")
            return None, None, None
        
        # Convert to numpy array
        portfolio_values = np.array(portfolio_values)
        
        # Calculate percentage returns relative to the initial value
        initial_value = portfolio_values[0]
        if initial_value <= 0:
            print("Warning: Initial portfolio value is zero or negative")
            return None, None, None
            
        historical_returns = [(value / initial_value - 1) * 100 for value in portfolio_values]
        
        # Get the dates as a list
        historical_dates = price_df.index.tolist()
        
        # Get the number of days
        historical_days = np.arange(len(historical_returns))
        
        return historical_days, historical_returns, historical_dates
    
    except Exception as e:
        print(f"Error in get_historical_portfolio_returns: {e}")
        return None, None, None

# Create a plotly chart for the simulation results
def create_simulation_chart(base_results, current_results=None):
    # Create a static chart first (without animation)
    fig = go.Figure()
    
    days = np.arange(252)
    
    # Add base expected value
    base_stats = base_results['portfolio_stats']
    
    # Add base expected value
    fig.add_trace(go.Scatter(
        x=days,
        y=base_stats['expected_value'],
        mode='lines',
        line=dict(color='rgba(70, 130, 180, 0.8)', width=2, dash='dash'),
        name='Expected Value (Base)'
    ))
    
    # If we have current results (with events/actions), add them
    if current_results:
        current_stats = current_results['portfolio_stats']
        
        # Add current expected value
        fig.add_trace(go.Scatter(
            x=days,
            y=current_stats['expected_value'],
            mode='lines',
            line=dict(color='rgba(46, 139, 87, 1)', width=3),
            name='Expected Value (With Events/Actions)'
        ))
        
        # Mark events with vertical lines
        for event in current_results['event_log']:
            day = event['day']
            fig.add_shape(
                type="line",
                x0=day, y0=0,
                x1=day, y1=current_stats['expected_value'][day] * 1.1,
                line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dot")
            )
            fig.add_annotation(
                x=day,
                y=current_stats['expected_value'][day] * 1.1,
                text=event['type'],
                showarrow=False,
                textangle=-90,
                font=dict(size=10)
            )
        
        # Mark actions with points
        for action in current_results['action_log']:
            day = action['day']
            fig.add_trace(go.Scatter(
                x=[day],
                y=[current_stats['expected_value'][day]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name=action['type'],
                showlegend=False
            ))
            fig.add_annotation(
                x=day,
                y=current_stats['expected_value'][day] * 0.95,
                text=action['type'],
                showarrow=False,
                textangle=-90,
                font=dict(size=10)
            )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Value Simulation (1 Year)',
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value (₹)',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add month markers on x-axis
    month_positions = [21*i for i in range(13)]  # Approximately 21 trading days per month
    month_labels = ['Start'] + [f'Month {i+1}' for i in range(12)]
    
    fig.update_xaxes(
        tickvals=month_positions,
        ticktext=month_labels
    )
    
    # Format y-axis as currency
    fig.update_yaxes(
        tickprefix='₹',
        separatethousands=True
    )
    
    return fig

# Function to generate investment personality using DeepSeek model
def generate_investment_personality(portfolio):
    """
    Generate a concise investment personality analysis (100-150 words) based on the user's portfolio
    using the DeepSeek reasoning model.
    
    Args:
        portfolio (dict): The user's portfolio data
        
    Returns:
        str: A concise investment personality analysis
    """
    # Create a system prompt that explains the task
    system_prompt = """
    You are a financial analyst specializing in investor psychology and portfolio analysis. 
    Your task is to analyze a portfolio and generate a CONCISE investment personality profile.
    
    Create a brief personality profile with the following structure:
    
    RISK PROFILE
    • [One-line statement about risk tolerance: conservative, moderate, or aggressive]
    • [One-line statement about investment style: value, growth, income, or blend]
    • [One-line statement about time horizon or decision-making speed]
    
    SECTOR STRATEGY
    • [One-line statement about sector preferences]
    • [One-line statement about diversification approach]
    • [One-line statement about sector allocation strategy]
    
    STRENGTHS & BLIND SPOTS
    • [One-line statement about key investment strength]
    • [One-line statement about potential blind spot]
    • [One-line statement about improvement opportunity]
    
    Important rules:
    - Use concise bullet points (one line each) as shown above
    - DO NOT mention specific stock names from the portfolio in your analysis
    - Use professional financial terminology while keeping the analysis accessible
    - Focus on the investor's personality traits and decision-making patterns
    - DO NOT include a word count at the end of your response
    - DO NOT include any disclaimers or additional notes
    - Keep the overall analysis under 150 words
    """
    
    # Create a user prompt with the portfolio information
    user_prompt = f"""
    Please analyze this investment portfolio and create a detailed investor personality profile:
    
    Portfolio:
    {json.dumps(portfolio, indent=2)}
    
    Based on this portfolio composition, what does it reveal about the investor's personality,
    risk tolerance, investment style, and decision-making approach?
    """
    
    try:
        # Call the DeepSeek model to generate the personality
        response = chat_with_deepseek(user_prompt, model="deepseek-r1", system_prompt=system_prompt)
        return response
    except Exception as e:
        # Fallback to a basic analysis if the model call fails
        print(f"Error generating investment personality: {e}")
        return "Unable to generate investment personality analysis. Please try again later."

# Function to generate custom scenario using LLM
def generate_custom_scenario(scenario_description, portfolio):
    # Create a system prompt that explains the task
    system_prompt = """
    You are a financial analyst assistant. Your task is to convert a user's scenario description into a structured event or action for a portfolio simulation.
    
    The output should be a JSON object with the following structure for an EVENT:
    {
        "id": "unique_id",
        "type": "event_type", (must be one of: market_correction, rbi_rate_cut, bull_run, bear_run, economic_downturn, us_tariff_increase, us_tariff_decrease, tech_sector_boom, tech_sector_crash, energy_sector_boom, energy_sector_crash, financial_sector_crisis)
        "title": "Event Title",
        "description": "Detailed description of the event",
        "icon": "emoji", (choose an appropriate emoji)
        "probability": "High/Medium/Low",
        "month": number (1-12)
    }
    
    OR for an ACTION:
    {
        "id": "unique_id",
        "type": "action_type", (must be one of: add_lumpsum_all, add_lumpsum_stock, book_profit, close_position)
        "title": "Action Title",
        "description": "Detailed description of the action",
        "icon": "emoji", (choose an appropriate emoji)
        "month": number (1-12),
        ... additional fields based on action type:
        - For add_lumpsum_all: "amount": number
        - For add_lumpsum_stock: "ticker": "stock_ticker", "amount": number
        - For book_profit: "ticker": "stock_ticker", "percentage": number
        - For close_position: "ticker": "stock_ticker"
    }
    
    Determine whether the user's description is an event (market condition) or an action (portfolio change).
    Choose the most appropriate event_type or action_type from the options listed.
    Set a reasonable month (1-12) when this should occur.
    For actions involving specific stocks, use one of these tickers from the portfolio: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, SBIN.NS

    If its a positive scenario, than map it to one of the events which can boost market returns such as : rbi_rate_cut, bull_run,, us_tariff_decrease, tech_sector_boom, energy_sector_boom
    If its a negative scenario, then map it to one of the events which can reduce market returns such as : tech_sector_crash, bear_run, economic_downturn, us_tariff_increase, market_correction, energy_sector_crash, financial_sector_crisis
    
    Return ONLY the JSON object without any additional text or explanation.
    """
    
    # Create a user prompt with the scenario description and portfolio information
    user_prompt = f"""
    Portfolio: {json.dumps(portfolio, indent=2)}
    
    Scenario description: {scenario_description}
    
    Convert this scenario into either an event or action JSON object as specified.
    """
    
    # Add debug output
    st.write("Comprehending your custom scenario...")
    
    # Call the LLM to generate the scenario
    try:
        # Debug: Show that we're calling the API
        # st.write("Calling DeepSeek API...")
        response = chat_with_deepseek(user_prompt, model="deepseek-r1", system_prompt=system_prompt)
        
        # Debug: Show the raw response
        # st.write("Raw API response:", response)
        
        # Parse the JSON response
        # First, try to find JSON within the response if it's not a pure JSON response
        import re
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            json_str = json_match.group(1)
            # st.write("Extracted JSON:", json_str)
        else:
            json_str = response
            # st.write("Using full response as JSON")
            
        # Parse the JSON
        try:
            scenario = json.loads(json_str)
            # st.write("Parsed JSON successfully")
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}")
            return None, f"JSON parsing error: {str(e)}"
        
        # Validate the scenario has required fields
        required_fields = ["id", "type", "title", "description", "icon", "month"]
        for field in required_fields:
            if field not in scenario:
                st.error(f"Generated scenario missing required field: {field}")
                return None, f"Generated scenario missing required field: {field}"
        
        # Determine if it's an event or action
        if scenario["type"] in ["market_correction", "rbi_rate_cut", "bull_run", "bear_run", 
                               "economic_downturn", "us_tariff_increase", "us_tariff_decrease", 
                               "tech_sector_boom", "tech_sector_crash", "energy_sector_boom", 
                               "energy_sector_crash", "financial_sector_crisis"]:
            scenario_type = "event"
            # Add probability if not present
            if "probability" not in scenario:
                scenario["probability"] = "Medium"
            st.write(f"Detected as EVENT: {scenario['type']}")
        else:
            scenario_type = "action"
            # Validate action-specific fields
            if scenario["type"] == "add_lumpsum_all" and "amount" not in scenario:
                st.error("Amount missing for add_lumpsum_all action")
                return None, "Amount missing for add_lumpsum_all action"
            elif scenario["type"] == "add_lumpsum_stock" and ("ticker" not in scenario or "amount" not in scenario):
                st.error("Ticker or amount missing for add_lumpsum_stock action")
                return None, "Ticker or amount missing for add_lumpsum_stock action"
            elif scenario["type"] == "book_profit" and ("ticker" not in scenario or "percentage" not in scenario):
                st.error("Ticker or percentage missing for book_profit action")
                return None, "Ticker or percentage missing for book_profit action"
            elif scenario["type"] == "close_position" and "ticker" not in scenario:
                st.error("Ticker missing for close_position action")
                return None, "Ticker missing for close_position action"
            st.write(f"Detected as ACTION: {scenario['type']}")
        
        st.success("Successfully comprehended your custom scenario!")
        return scenario, scenario_type
    
    except Exception as e:
        st.error(f"Error generating scenario: {str(e)}")
        return None, f"Error generating scenario: {str(e)}"

# Main app
def main():
    st.markdown("<h1 class='header' style='margin-top: -70px;'>RIZQ - Hyper-personalized Wealth Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader' style='color: white;'>Your Wealth, With a Sixth Sense.</p>", unsafe_allow_html=True)
    
    # Add CSS for personality card that will be used later
    st.markdown("""
    <style>
    .personality-card {
        background: linear-gradient(90deg, #1E293B, #2D3748);
        border-radius: 12px;
        padding: 25px;
        margin: -50px 0 30px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .personality-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0) 30%, rgba(255,255,255,0.05), rgba(255,255,255,0) 70%);
        transform: translateX(-100%);
        animation: shine 4s infinite;
    }
    
    .personality-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 15px;
    }
    
    .personality-icon {
        font-size: 2.5em;
        margin-right: 20px;
        color: #4299E1;
    }
    
    .personality-title {
        font-size: 1.4em;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 5px;
    }
    
    .personality-subtitle {
        color: #A0AEC0;
        font-size: 0.9em;
    }
    
    .personality-content {
        color: #CBD5E0;
        font-size: 0.95em;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add custom styling for the fetch portfolio button
    st.markdown("""
    <style>
    .fetch-portfolio-btn {
        background: linear-gradient(90deg, #2B6CB0, #4CAF50);
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 1.2em;
        margin: 40px auto;
        display: block;
        width: 80%;
        max-width: 500px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .fetch-portfolio-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }
    
    .fetch-portfolio-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 60px 0;
    }
    
    .fetch-portfolio-icon {
        font-size: 4em;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #2B6CB0, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add horizontal card showing percentile comparison
    st.markdown("""
    <style>
    .percentile-card {
        background: linear-gradient(90deg, #1E293B, #2D3748);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        position: relative;
        overflow: hidden;
    }
    
    .percentile-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0) 30%, rgba(255,255,255,0.05), rgba(255,255,255,0) 70%);
        transform: translateX(-100%);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        20% { transform: translateX(100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Animation for feature cards */
    @keyframes cardShine {
        0% { transform: translateX(-100%); }
        20% { transform: translateX(100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Feature card styling */
    .feature-card {
        position: relative;
        overflow: hidden;
    }
    
    .shine-effect {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0) 30%, rgba(255,255,255,0.05), rgba(255,255,255,0) 70%);
        transform: translateX(-100%);
        animation: cardShine 4s infinite;
        z-index: 1;
    }
    
    /* Add JavaScript to ensure animations work */
    </style>
    
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Function to apply shine effect to feature cards
        function applyShineEffect() {
            const featureCards = document.querySelectorAll('.feature-card');
            featureCards.forEach((card, index) => {
                const shineEffect = card.querySelector('.shine-effect');
                if (shineEffect) {
                    shineEffect.style.animationDelay = (index * 1) + 's';
                }
            });
        }
        
        // Call the function when DOM is loaded
        applyShineEffect();
        
        // Also call it after a short delay to ensure Streamlit has fully rendered
        setTimeout(applyShineEffect, 1000);
    });
    </script>
    
    <style>
    
    .percentile-icon {
        font-size: 2.5em;
        margin-right: 20px;
        color: #4299E1;
    }
    
    .percentile-content {
        flex: 1;
    }
    
    .percentile-title {
        font-size: 1.2em;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 5px;
    }
    
    .percentile-description {
        color: #A0AEC0;
        font-size: 0.9em;
    }
    
    .percentile-badge {
        background: linear-gradient(90deg, #4299E1, #38A169);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1em;
        margin-left: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'portfolio_fetched' not in st.session_state:
        st.session_state.portfolio_fetched = False
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    
    # Function to handle button click
    def handle_fetch_portfolio():
        st.session_state.button_clicked = True
    
    # Show fetch portfolio button if portfolio not yet fetched
    if not st.session_state.portfolio_fetched:
        # Add some information about what the app does with dark theme
        st.markdown("""
        <div style="margin-top: -50px; padding: 30px; border-radius: 12px; background-color: #1E293B; text-align: center; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);">
            <h3 style="color: #E2E8F0; margin-bottom: 20px;">What you can do with RIZQ!</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; margin-top: 20px;">
                <div class="feature-card" style="flex: 1; min-width: 200px; margin: 10px; padding: 20px; background-color: #263850; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-left: 4px solid #4299E1; position: relative; overflow: hidden;">
                    <div class="shine-effect"></div>
                    <div style="font-size: 2.5em; margin-bottom: 15px; color: #63B3ED;">🔮</div>
                    <h4 style="color: #E2E8F0; margin-bottom: 10px;">Predict Future Performance</h4>
                    <p style="color: #A0AEC0; line-height: 1.5;">See how your portfolio might perform over the next year using Monte Carlo simulations</p>
                </div>
                <div class="feature-card" style="flex: 1; min-width: 200px; margin: 10px; padding: 20px; background-color: #263850; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-left: 4px solid #F56565; position: relative; overflow: hidden;">
                    <div class="shine-effect" style="animation-delay: 1s;"></div>
                    <div style="font-size: 2.5em; margin-bottom: 15px; color: #FC8181;">📈</div>
                    <h4 style="color: #E2E8F0; margin-bottom: 10px;">Test Market Events</h4>
                    <p style="color: #A0AEC0; line-height: 1.5;">Simulate how different market events like corrections or sector booms might affect your investments</p>
                </div>
                <div class="feature-card" style="flex: 1; min-width: 200px; margin: 10px; padding: 20px; background-color: #263850; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-left: 4px solid #38A169; position: relative; overflow: hidden;">
                    <div class="shine-effect" style="animation-delay: 2s;"></div>
                    <div style="font-size: 2.5em; margin-bottom: 15px; color: #68D391;">💰</div>
                    <h4 style="color: #E2E8F0; margin-bottom: 10px;">Plan Investment Actions</h4>
                    <p style="color: #A0AEC0; line-height: 1.5;">Test different investment strategies like adding lumpsum amounts or booking profits</p>
                </div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 10px; margin-bottom: 10px;">
            <div style="width: 100%; max-width: 400px; text-align: center;">
        """, unsafe_allow_html=True)
        
        # Add custom styling for the fetch portfolio button specifically
        st.markdown("""
        <style>
        /* Target specifically the fetch portfolio button */
        div[data-testid="stButton"] > button[kind="primaryFormSubmit"],
        div[data-testid="stButton"] > button[data-testid="baseButton-primaryFormSubmit"],
        div[data-testid="stButton"] button:first-of-type {
            background: linear-gradient(90deg, #2B6CB0, #38A169) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            width: 100% !important;
            font-size: 1.1em !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
            margin: 0 auto !important;
            display: block !important;
        }
        
        div[data-testid="stButton"] > button[kind="primaryFormSubmit"]:hover,
        div[data-testid="stButton"] > button[data-testid="baseButton-primaryFormSubmit"]:hover,
        div[data-testid="stButton"] button:first-of-type:hover {
            background: linear-gradient(90deg, #3182CE, #48BB78) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3) !important;
            color: #F0FFF4 !important; /* Sleek mint color that complements the gradient */
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        }
        
        div[data-testid="stButton"] > button[kind="primaryFormSubmit"]:active,
        div[data-testid="stButton"] > button[data-testid="baseButton-primaryFormSubmit"]:active,
        div[data-testid="stButton"] button:first-of-type:active {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a centered button below the info section
        if not st.session_state.button_clicked:
            fetch_clicked = st.button("Fetch my Portfolio from my Demat", key="fetch_portfolio_btn", on_click=handle_fetch_portfolio)
        
        # Only show spinner and process if button is clicked
        if st.session_state.button_clicked:
            # Show a spinner while "fetching" the portfolio
            with st.spinner("Connecting to your demat account..."):
                # Simulate a delay to make it feel like it's fetching data
                time.sleep(2)
                # Set the state and rerun immediately
                st.session_state.portfolio_fetched = True
                st.rerun()  # Using rerun to ensure immediate refresh
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Return early to not show the rest of the app
        return
    
    # If we get here, the portfolio has been fetched, so show the rest of the app
    
    # Get portfolio data
    portfolio = get_default_portfolio()
    
    # Generate investment personality if not already in session state
    if 'investment_personality' not in st.session_state:
        st.write("✅ Successfully fetched your portfolio from your demat account.")
        with st.spinner("Analyzing your investment personality..."):
            st.session_state.investment_personality = generate_investment_personality(portfolio)
    
    # Display investment personality analysis
    st.markdown(f"""
    <div class="personality-card">
        <div class="personality-header">
            <div class="personality-icon">🧍</div>
            <div>
                <div class="personality-title">Your Investment Personality</div>
                <div class="personality-subtitle">AI-powered analysis of your investment style and preferences</div>
            </div>
        </div>
        <div class="personality-content">
            {st.session_state.investment_personality}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Run base simulation (without events or actions)
    base_results = run_base_simulation(portfolio)
    
    # Initialize session state for selected events and actions
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = []
    if 'selected_actions' not in st.session_state:
        st.session_state.selected_actions = []
    
    # Function to calculate percentile of returns compared to average investors
    def calculate_return_percentile(return_value):
        # Static data for average investor returns (realistic numbers)
        # These represent annual returns for different percentiles
        investor_returns = {
            10: 2.5,   # Bottom 10% of investors achieve 2.5% or less
            25: 5.0,   # Bottom 25% of investors achieve 5.0% or less
            50: 8.0,   # Median investor achieves 8.0%
            75: 12.0,  # Top 25% of investors achieve 12.0% or more
            90: 16.0,  # Top 10% of investors achieve 16.0% or more
            95: 20.0,  # Top 5% of investors achieve 20.0% or more
            99: 25.0   # Top 1% of investors achieve 25.0% or more
        }
        
        # Find which percentile the user's return falls into
        if return_value <= investor_returns[10]:
            return "bottom 10%"
        elif return_value <= investor_returns[25]:
            return "bottom 25%"
        elif return_value <= investor_returns[50]:
            return "average 50%"
        elif return_value <= investor_returns[75]:
            return "top 25%"
        elif return_value <= investor_returns[90]:
            return "top 10%"
        elif return_value <= investor_returns[95]:
            return "top 5%"
        else:
            return "top 1%"

    st.write("")
    st.write("")
    st.markdown("<h1 class='header'>Portfolio Summary</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Quick stats and potential future projection of your portfolio</p>", unsafe_allow_html=True)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    initial_value = base_results['portfolio_stats']['initial_value']
    expected_final_value = base_results['portfolio_stats']['final_expected_value']
    expected_return = base_results['portfolio_stats']['expected_return']
    volatility = base_results['portfolio_stats']['volatility']
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Initial Value</div>
            <div class='metric-value'>{format_currency(initial_value)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Expected Final Value</div>
            <div class='metric-value'>{format_currency(expected_final_value)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        return_class = "positive" if expected_return > 0 else "negative"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Expected Return</div>
            <div class='metric-value {return_class}'>{expected_return:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Portfolio Volatility</div>
            <div class='metric-value'>{volatility:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Calculate percentile for the user's expected return
    percentile_category = calculate_return_percentile(expected_return)
    
    
    # Determine icon and message based on percentile
    if "top 1%" in percentile_category:
        icon = "🏆"
        message = "Exceptional! Your portfolio is outperforming 99% of retail investors."
    elif "top 5%" in percentile_category:
        icon = "🌟"
        message = "Outstanding! Your portfolio is outperforming 95% of retail investors."
    elif "top 10%" in percentile_category:
        icon = "🚀"
        message = "Excellent! Your portfolio is outperforming 90% of retail investors."
    elif "top 25%" in percentile_category:
        icon = "📈"
        message = "Great job! Your portfolio is outperforming 75% of retail investors."
    elif "average" in percentile_category:
        icon = "⚖️"
        message = "Your portfolio is performing around the market average, in line with most retail investors."
    elif "bottom 25%" in percentile_category:
        icon = "📊"
        message = "Your portfolio has room for improvement. Consider diversifying or adjusting your strategy."
    else:
        icon = "📉"
        message = "Your portfolio is underperforming compared to most retail investors. Consider consulting a financial advisor."
    
    st.markdown(f"""
    <div class="percentile-card">
        <div class="percentile-icon">{icon}</div>
        <div class="percentile-content">
            <div class="percentile-title">Performance Comparison</div>
            <div class="percentile-description">{message}</div>
        </div>
        <div class="percentile-badge">
            {percentile_category.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2>Stock Details</h2>", unsafe_allow_html=True)
        
    # Get base results for comparison
    base_sim_results = {}
    for ticker in portfolio.keys():
        base_sim_results[ticker] = base_results['stock_simulations'][ticker]
    
    # Create a DataFrame for the portfolio
    portfolio_data = []
    for ticker, details in portfolio.items():
        # Get simulation results for this stock
        base_sim_result = base_sim_results[ticker]
        
        # Get prices
        initial_price = base_sim_result['expected_price'][0]
        base_final_price = base_sim_result['expected_price'][-1]
        
        # Calculate values
        initial_value = details['quantity'] * initial_price
        base_final_value = details['quantity'] * base_final_price
        
        # Calculate returns accounting for additional investments
        adjusted_initial_value = initial_value
        if adjusted_initial_value > 0:
            base_return_pct = (base_final_value / adjusted_initial_value - 1) * 100
        else:
            base_return_pct = 0
        
        # Get sector badge class
        sector = details.get('sector', 'Unknown')
        sector_class = ""
        if sector == "Technology":
            sector_class = "badge-tech"
        elif sector == "Financial":
            sector_class = "badge-finance"
        elif sector == "Energy":
            sector_class = "badge-energy"
        
        # Format returns with color
        return_class = "positive" if base_return_pct > 0 else "negative"
        
        portfolio_data.append({
            "Ticker": ticker,
            "Sector": f"<span class='badge {sector_class}'>{sector}</span>",
            "Quantity": details['quantity'],
            "Buy Price": details['buy_price'],
            "Current Price": round(initial_price, 2),
            "Expected Final Price": round(base_final_price, 2),
            "Expected Return": f"<span class='{return_class}'>{base_return_pct:.2f}%</span>",
            "Initial Value": format_currency(initial_value),
            "Expected Final Value": format_currency(base_final_value)
        })
    
    # Display portfolio details table
    df = pd.DataFrame(portfolio_data)
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Create two columns for the pie charts
    st.markdown("<h3>Portfolio Allocation</h3>", unsafe_allow_html=True)
    col_stock, col_sector = st.columns(2)
    
    # Calculate stock allocation
    stock_values = {}
    total_value = 0
    for ticker, details in portfolio.items():
        initial_price = base_sim_results[ticker]['expected_price'][0]
        value = details['quantity'] * initial_price
        stock_values[ticker.replace('.NS', '')] = value
        total_value += value
    
    # Create stock allocation pie chart
    with col_stock:
        st.markdown("<h4 style='text-align: center;'>Stock Allocation</h4>", unsafe_allow_html=True)
        
        stock_labels = list(stock_values.keys())
        stock_values_list = list(stock_values.values())
        
        # Define stock colors - using a gradient of blues
        stock_colors = [
            '#2B6CB0',  # RELIANCE
            '#3182CE',  # TCS
            '#4299E1',  # HDFCBANK
            '#63B3ED',  # INFY
            '#90CDF4',  # SBIN
        ]
        
        # Create the stock allocation pie chart
        fig_stock = go.Figure(data=[go.Pie(
            labels=stock_labels,
            values=stock_values_list,
            hole=.4,
            marker=dict(colors=stock_colors),
            textinfo='label+percent',
            textposition='outside',
            pull=[0.05 if stock == max(stock_values, key=stock_values.get) else 0 for stock in stock_labels],
            hoverinfo='label+value+percent',
            hovertemplate='%{label}<br>₹%{value:,.2f}<br>%{percent}'
        )])
        
        fig_stock.update_layout(
            height=450,  # Increased height to accommodate legend
            margin=dict(l=20, r=20, t=20, b=60),  # Increased bottom margin
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,  # Moved legend further down
                xanchor="center",
                x=0.5
            ),
            annotations=[dict(
                text=f'Total Value:<br>{format_currency(total_value)}',
                showarrow=False,
                font=dict(size=14),
                x=0.5,
                y=0.5
            )]
        )
        
        # Display the stock allocation chart
        st.plotly_chart(fig_stock, use_container_width=True)
    
    # Calculate sector allocation
    with col_sector:
        st.markdown("<h4 style='text-align: center;'>Sector Allocation</h4>", unsafe_allow_html=True)
        
        sector_values = {}
        for ticker, details in portfolio.items():
            sector = details.get('sector', 'Unknown')
            initial_price = base_sim_results[ticker]['expected_price'][0]
            value = details['quantity'] * initial_price
            
            if sector in sector_values:
                sector_values[sector] += value
            else:
                sector_values[sector] = value
        
        # Create sector pie chart
        sector_labels = list(sector_values.keys())
        sector_values_list = list(sector_values.values())
        
        # Define sector colors
        sector_colors = {
            'Technology': '#4299E1',  # Blue
            'Financial': '#805AD5',   # Purple
            'Energy': '#F6AD55',      # Orange
            'Unknown': '#A0AEC0'      # Gray
        }
        
        # Get colors for each sector
        colors = [sector_colors.get(sector, '#A0AEC0') for sector in sector_labels]
        
        # Create the sector allocation pie chart
        fig_sector = go.Figure(data=[go.Pie(
            labels=sector_labels,
            values=sector_values_list,
            hole=.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='outside',
            pull=[0.05 if sector == max(sector_values, key=sector_values.get) else 0 for sector in sector_labels],
            hoverinfo='label+value+percent',
            hovertemplate='%{label}<br>₹%{value:,.2f}<br>%{percent}'
        )])
        
        fig_sector.update_layout(
            height=450,  # Increased height to accommodate legend
            margin=dict(l=20, r=20, t=20, b=60),  # Increased bottom margin
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,  # Moved legend further down
                xanchor="center",
                x=0.5
            ),
            annotations=[dict(
                text=f'Total Value:<br>{format_currency(sum(sector_values_list))}',
                showarrow=False,
                font=dict(size=14),
                x=0.5,
                y=0.5
            )]
        )
        
        # Display the sector allocation chart
        st.plotly_chart(fig_sector, use_container_width=True)

    st.write("")
    st.write("")
    st.write("")
    st.markdown("<h1 class='header'>Portfolio Simulation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Stress test your portfolio over the next year with potential market events and investment actions</p>", unsafe_allow_html=True)

    # Run simulation with selected events and actions
    current_results = None
    if st.session_state.selected_events or st.session_state.selected_actions:
        current_results = run_simulation_with_selections(
            portfolio, 
            st.session_state.selected_events, 
            st.session_state.selected_actions
        )
    
    # Fetch historical data for the portfolio
    with st.spinner("Fetching historical portfolio data..."):
        hist_days, hist_returns, hist_dates = get_historical_portfolio_returns(portfolio)
    
    # Create chart with built-in animation showing percentage returns
    days = np.arange(252)
    base_stats = base_results['portfolio_stats']
    
    # Calculate percentage returns from initial value
    initial_value = base_stats['initial_value']
    base_pct_returns = [(value / initial_value - 1) * 100 for value in base_stats['expected_value']]
    
    # Check if percentile data is available, otherwise calculate approximations
    base_pct_5th = [(value / initial_value - 1) * 100 for value in base_stats['min_value']]
    base_pct_95th = [(value / initial_value - 1) * 100 for value in base_stats['max_value']]
    
    # Create a figure with animation frames
    fig = go.Figure()
    
    # Add historical data if available
    if hist_days is not None and hist_returns is not None:
        # Create a continuous x-axis by shifting the forecast days
        # Historical days will be negative (counting back from today)
        hist_len = len(hist_days)
        shifted_hist_days = [-hist_len + i for i in range(hist_len)]
        
        # Normalize historical returns to end at 0% at today
        # This ensures continuity between historical and future data
        last_hist_return = hist_returns[-1]
        normalized_hist_returns = [(ret - last_hist_return) for ret in hist_returns]
        
        # Add historical returns trace
        fig.add_trace(go.Scatter(
            x=shifted_hist_days,
            y=normalized_hist_returns,
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.8)', width=2),
            name='Historical Returns (6 Months)'
        ))
        
        # Add a vertical line at x=0 (today) to separate historical and forecast data
        min_y = min(min(base_pct_5th), min(normalized_hist_returns)) - 5
        max_y = max(max(base_pct_95th), max(normalized_hist_returns)) + 5
        
        fig.add_shape(
            type="line",
            x0=0, y0=min_y,
            x1=0, y1=max_y,
            line=dict(color="rgba(200, 200, 200, 0.7)", width=1.5, dash="dot")
        )
        
        # Add "Future Projection" text
        fig.add_annotation(
            x=0,  # Position at month 1
            y=max_y,
            text="Future Projection",
            showarrow=False,
            font=dict(size=12, color="rgba(70, 130, 180, 0.8)")
        )
    
    # Add 5th and 95th percentile boundaries as a filled area
    fig.add_trace(go.Scatter(
        x=days,
        y=base_pct_95th,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=base_pct_5th,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(70, 130, 180, 0.2)',
        name='90% Confidence Interval (Base)',
        hoverinfo='skip'
    ))
    
    # Add base expected value
    fig.add_trace(go.Scatter(
        x=days,
        y=base_pct_returns,
        mode='lines',
        line=dict(color='rgba(70, 130, 180, 0.8)', width=2, dash='dash'),
        name='Expected Return % (Base)'
    ))
    
    # If we have current results (with events/actions), add them
    if current_results:
        current_stats = current_results['portfolio_stats']
        
        # For the graph visualization, we need to handle lumpsum investments without causing spikes or dips
        
        # Get the portfolio values
        base_portfolio_values = base_stats['expected_value']
        current_portfolio_values = current_stats['expected_value']
        current_portfolio_5th = current_stats['min_value']
        current_portfolio_95th = current_stats['max_value']
        
        # Create a map of actions with their details
        action_map = {}
        
        # Process all selected actions to build the action map
        for action_id in st.session_state.selected_actions:
            # Check in predefined actions
            for action in get_actions():
                if action["id"] == action_id:
                    day = (action["month"] - 1) * 21 + 10  # Middle of the month
                    action_type = action["type"]
                    amount = action.get("amount", 0)
                    
                    if day not in action_map:
                        action_map[day] = []
                    
                    action_map[day].append({
                        "type": action_type,
                        "amount": amount,
                        "ticker": action.get("ticker", None),
                        "percentage": action.get("percentage", None)
                    })
                    break
            
            # Check in custom scenarios
            if 'custom_scenarios' in st.session_state:
                for custom_item in st.session_state.custom_scenarios:
                    scenario = custom_item["scenario"]
                    if scenario["id"] == action_id and custom_item["type"] == "action":
                        day = (scenario["month"] - 1) * 21 + 10
                        action_type = scenario["type"]
                        amount = scenario.get("amount", 0)
                        
                        if day not in action_map:
                            action_map[day] = []
                        
                        action_map[day].append({
                            "type": action_type,
                            "amount": amount,
                            "ticker": scenario.get("ticker", None),
                            "percentage": scenario.get("percentage", None)
                        })
                        break
            
            # Check in AI recommendations
            if 'ai_recommendations' in st.session_state:
                for ai_action in st.session_state.ai_recommendations:
                    if ai_action["id"] == action_id:
                        day = (ai_action["month"] - 1) * 21 + 10
                        action_type = ai_action["type"]
                        amount = ai_action.get("amount", 0)
                        
                        if day not in action_map:
                            action_map[day] = []
                        
                        action_map[day].append({
                            "type": action_type,
                            "amount": amount,
                            "ticker": ai_action.get("ticker", None),
                            "percentage": ai_action.get("percentage", None)
                        })
                        break
        
        # Sort action days
        action_days = sorted(list(action_map.keys()))
        
        # Calculate percentage returns with smooth transitions for actions
        if action_days:
            # Create arrays for cumulative capital adjustments
            capital_adjustments = np.zeros(252)
            
            # Calculate capital adjustments for each day
            for day in action_days:
                for action in action_map[day]:
                    if action["type"] == "add_lumpsum_all" or action["type"] == "add_lumpsum_stock":
                        # Add lumpsum investment
                        capital_adjustments[day:] += action["amount"]
                    elif action["type"] == "book_profit" and action["ticker"] and action["percentage"]:
                        # Calculate approximate value of sold shares
                        ticker = action["ticker"]
                        percentage = action["percentage"] / 100.0
                        for t, details in portfolio.items():
                            if t == ticker:
                                # Use current price as an approximation
                                price = base_sim_results[ticker]['expected_price'][0]
                                shares_sold = details['quantity'] * percentage
                                capital_adjustments[day:] -= shares_sold * price
                    elif action["type"] == "close_position" and action["ticker"]:
                        # Calculate value of closed position
                        ticker = action["ticker"]
                        for t, details in portfolio.items():
                            if t == ticker:
                                # Use current price as an approximation
                                price = base_sim_results[ticker]['expected_price'][0]
                                capital_adjustments[day:] -= details['quantity'] * price
            
            # Calculate adjusted initial values for each day
            initial_value = current_portfolio_values[0]
            adjusted_initial_values = np.ones(252) * initial_value
            
            # For each day, adjust the initial value based on cumulative capital adjustments
            for day in range(252):
                adjusted_initial_values[day] += capital_adjustments[day]
            
            # Calculate percentage returns using the adjusted initial values
            current_pct_returns = [(current_portfolio_values[day] / adjusted_initial_values[day] - 1) * 100 for day in range(252)]
            current_pct_5th = [(current_portfolio_5th[day] / adjusted_initial_values[day] - 1) * 100 for day in range(252)]
            current_pct_95th = [(current_portfolio_95th[day] / adjusted_initial_values[day] - 1) * 100 for day in range(252)]
        else:
            # If no actions, calculate returns normally
            current_pct_returns = [(value / current_portfolio_values[0] - 1) * 100 for value in current_portfolio_values]
            current_pct_5th = [(value / current_portfolio_values[0] - 1) * 100 for value in current_portfolio_5th]
            current_pct_95th = [(value / current_portfolio_values[0] - 1) * 100 for value in current_portfolio_95th]
        
        # Add 5th and 95th percentile boundaries for current scenario
        fig.add_trace(go.Scatter(
            x=days,
            y=current_pct_95th,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=days,
            y=current_pct_5th,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(46, 139, 87, 0.2)',
            name='90% Confidence Interval (With Events/Actions)',
            hoverinfo='skip'
        ))
        
        # Add current expected value
        fig.add_trace(go.Scatter(
            x=days,
            y=current_pct_returns,
            mode='lines',
            line=dict(color='rgba(46, 139, 87, 1)', width=3),
            name='Expected Return % (With Events/Actions)'
        ))
        
        # Mark events with vertical lines
        for event in current_results['event_log']:
            day = event['day']
            max_y_value = max(current_pct_returns[day], base_pct_returns[day]) * 1.1
            fig.add_shape(
                type="line",
                x0=day, y0=0,
                x1=day, y1=max_y_value,
                line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dot")
            )
            fig.add_annotation(
                x=day,
                y=max_y_value,
                text=event['type'],
                showarrow=False,
                textangle=-90,
                font=dict(size=10)
            )
        
        # Mark actions with points
        for action in current_results['action_log']:
            day = action['day']
            fig.add_trace(go.Scatter(
                x=[day],
                y=[current_pct_returns[day]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name=action['type'],
                showlegend=False
            ))
            fig.add_annotation(
                x=day,
                y=current_pct_returns[day] * 0.95,
                text=action['type'],
                showarrow=False,
                textangle=-90,
                font=dict(size=10)
            )
    
    # Create frames for animation
    frames = []
    
    # Add historical data to all frames if available
    historical_traces = []
    if hist_days is not None and hist_returns is not None:
        hist_len = len(hist_days)
        shifted_hist_days = [-hist_len + i for i in range(hist_len)]
        
        # Normalize historical returns to end at 0% at today
        # This ensures continuity between historical and future data
        last_hist_return = hist_returns[-1]
        normalized_hist_returns = [(ret - last_hist_return) for ret in hist_returns]
        
        # Historical data trace (this will be static in all frames)
        historical_trace = go.Scatter(
            x=shifted_hist_days,
            y=normalized_hist_returns,
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.8)', width=2),
            name='Historical Returns (6 Months)'
        )
        historical_traces.append(historical_trace)
    
    # Create animation frames for future projections
    for i in range(0, 252, 5):  # Create frames at intervals
        frame_data = historical_traces.copy()  # Start with historical data in each frame
        
        # Base 5th percentile for this frame
        if i > 0:
            base_5th_frame = go.Scatter(
                x=days[:i+1],
                y=base_pct_5th[:i+1],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
            frame_data.append(base_5th_frame)
            
            # Base 95th percentile for this frame
            base_95th_frame = go.Scatter(
                x=days[:i+1],
                y=base_pct_95th[:i+1],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(70, 130, 180, 0.2)',
                name='90% Confidence Interval (Base)',
                hoverinfo='skip'
            )
            frame_data.append(base_95th_frame)
        
        # Base value for this frame
        base_frame = go.Scatter(
            x=days[:i+1],
            y=base_pct_returns[:i+1],
            mode='lines',
            line=dict(color='rgba(70, 130, 180, 0.8)', width=2, dash='dash'),
            name='Expected Return % (Base)'
        )
        frame_data.append(base_frame)
        
        # Current value for this frame (if available)
        if current_results and i > 0:
            # Current 5th percentile for this frame
            current_5th_frame = go.Scatter(
                x=days[:i+1],
                y=current_pct_5th[:i+1],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
            frame_data.append(current_5th_frame)
            
            # Current 95th percentile for this frame
            current_95th_frame = go.Scatter(
                x=days[:i+1],
                y=current_pct_95th[:i+1],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(46, 139, 87, 0.2)',
                name='90% Confidence Interval (With Events/Actions)',
                hoverinfo='skip'
            )
            frame_data.append(current_95th_frame)
            
            # Current expected value for this frame
            current_frame = go.Scatter(
                x=days[:i+1],
                y=current_pct_returns[:i+1],
                mode='lines',
                line=dict(color='rgba(46, 139, 87, 1)', width=3),
                name='Expected Return % (With Events/Actions)'
            )
            frame_data.append(current_frame)
        
        frames.append(go.Frame(data=frame_data, name=f"frame{i}"))
    
    fig.frames = frames
    
    # Update layout
    fig.update_layout(
        title='Portfolio Return Simulation (Past 6 Months & Future 1 Year)',
        xaxis_title='Trading Days',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}]
                }
            ],
            'x': 0,
            'y': -0.1,
            'xanchor': 'right',
            'yanchor': 'top'
        }]
    )
    
    # Add month markers on x-axis
    # For historical data (negative days)
    if hist_days is not None:
        hist_len = len(hist_days)
        # Divide historical period into 6 parts, but exclude the last point (which would be -0 mo)
        hist_month_positions = [-hist_len + (hist_len // 6) * i for i in range(6)]  # Only include -6 mo to -1 mo
        hist_month_labels = [f"-{6-i} mo" for i in range(6)]  # Label as "-6 mo", "-5 mo", etc.
        
        # For future data (positive days)
        future_month_positions = [21*i for i in range(13)]  # Approximately 21 trading days per month
        future_month_labels = ['Today'] + [f"+{i+1} mo" for i in range(12)]  # Label as "+1 mo", "+2 mo", etc.
        
        # Combine positions and labels
        all_positions = hist_month_positions + future_month_positions
        all_labels = hist_month_labels + future_month_labels
    else:
        # If no historical data, just use future months
        all_positions = [21*i for i in range(13)]  # Approximately 21 trading days per month
        all_labels = ['Start'] + [f'Month {i+1}' for i in range(12)]
    
    fig.update_xaxes(
        tickvals=all_positions,
        ticktext=all_labels
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(
        ticksuffix='%'
    )
    
    # Add a horizontal line at y=0 for reference
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=251, y1=0,
        line=dict(color="rgba(0, 0, 0, 0.3)", width=1, dash="dot")
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True, key="animated_chart")
    
    # Add JavaScript to auto-play the animation
    st.markdown("""
    <script>
        // Function to find and click the play button
        function clickPlayButton() {
            console.log("Attempting to find and click play button...");
            
            // Try different selectors to find the play button
            const selectors = [
                'button.modebar-btn[data-title="Play"]',
                'button.modebar-btn[title="Play"]',
                'a.modebar-btn[data-title="Play"]',
                'a.modebar-btn[title="Play"]'
            ];
            
            let buttonFound = false;
            
            // Try each selector
            for (const selector of selectors) {
                const buttons = document.querySelectorAll(selector);
                if (buttons.length > 0) {
                    console.log(`Found play button with selector: ${selector}`);
                    buttons[0].click();
                    buttonFound = true;
                    break;
                }
            }
            
            // If no button found with specific selectors, try the generic approach
            if (!buttonFound) {
                console.log("Trying generic approach to find play button...");
                const allButtons = document.querySelectorAll('button.modebar-btn, a.modebar-btn');
                for (let button of allButtons) {
                    const title = button.getAttribute('data-title') || button.getAttribute('title') || '';
                    if (title.includes('Play') || button.innerHTML.includes('Play')) {
                        console.log("Found play button via generic approach");
                        button.click();
                        buttonFound = true;
                        break;
                    }
                }
            }
            
            return buttonFound;
        }
        
        // Try multiple times with increasing delays
        let attempts = 0;
        const maxAttempts = 5;
        
        function attemptClickWithDelay() {
            attempts++;
            console.log(`Attempt ${attempts} to click play button`);
            
            if (clickPlayButton()) {
                console.log("Successfully clicked play button");
                return;
            }
            
            if (attempts < maxAttempts) {
                // Exponential backoff for retry timing
                const delay = 1000 * Math.pow(1.5, attempts);
                console.log(`Will try again in ${delay}ms`);
                setTimeout(attemptClickWithDelay, delay);
            } else {
                console.log("Failed to find play button after maximum attempts");
            }
        }
        
        // Start the first attempt after initial delay
        setTimeout(attemptClickWithDelay, 2000);
        
        // Also try when the window is fully loaded
        window.addEventListener('load', function() {
            setTimeout(attemptClickWithDelay, 1000);
        });
        
        // And try when any iframe is loaded (in case the chart is in an iframe)
        document.querySelectorAll('iframe').forEach(iframe => {
            iframe.addEventListener('load', function() {
                setTimeout(attemptClickWithDelay, 1000);
            });
        });
    </script>
    """, unsafe_allow_html=True)
    
    # Update the metrics based on current results if available
    if current_results:
        current_initial_value = current_results['portfolio_stats']['initial_value']
        current_final_value = current_results['portfolio_stats']['final_expected_value']
        current_volatility = current_results['portfolio_stats']['volatility']
        
        # Calculate adjustments for capital added to or removed from the portfolio
        capital_adjustment = 0  # Positive for additions, negative for removals
        for action_id in st.session_state.selected_actions:
            # Check in predefined actions
            action_found = False
            for action in get_actions():
                if action["id"] == action_id:
                    if action["type"] == "add_lumpsum_all":
                        # Add lumpsum investment to all stocks
                        capital_adjustment += action["amount"]
                    elif action["type"] == "add_lumpsum_stock":
                        # Add lumpsum investment to specific stock
                        capital_adjustment += action["amount"]
                    elif action["type"] == "book_profit":
                        # Calculate approximate value of sold shares
                        ticker = action["ticker"]
                        percentage = action["percentage"] / 100.0
                        for t, details in portfolio.items():
                            if t == ticker:
                                # Use current price as an approximation
                                price = base_sim_results[ticker]['expected_price'][0]
                                shares_sold = details['quantity'] * percentage
                                capital_adjustment -= shares_sold * price
                    elif action["type"] == "close_position":
                        # Calculate value of closed position
                        ticker = action["ticker"]
                        for t, details in portfolio.items():
                            if t == ticker:
                                # Use current price as an approximation
                                price = base_sim_results[ticker]['expected_price'][0]
                                capital_adjustment -= details['quantity'] * price
                    action_found = True
                    break
            
            # Also check in custom scenarios and AI recommendations
            if not action_found:
                # Check in custom scenarios
                if 'custom_scenarios' in st.session_state:
                    for custom_item in st.session_state.custom_scenarios:
                        scenario = custom_item["scenario"]
                        if scenario["id"] == action_id and custom_item["type"] == "action":
                            if scenario["type"] == "add_lumpsum_all" and "amount" in scenario:
                                capital_adjustment += scenario["amount"]
                            elif scenario["type"] == "add_lumpsum_stock" and "amount" in scenario:
                                capital_adjustment += scenario["amount"]
                            elif scenario["type"] == "book_profit" and "ticker" in scenario and "percentage" in scenario:
                                ticker = scenario["ticker"]
                                percentage = scenario["percentage"] / 100.0
                                for t, details in portfolio.items():
                                    if t == ticker:
                                        price = base_sim_results[ticker]['expected_price'][0]
                                        shares_sold = details['quantity'] * percentage
                                        capital_adjustment -= shares_sold * price
                            elif scenario["type"] == "close_position" and "ticker" in scenario:
                                ticker = scenario["ticker"]
                                for t, details in portfolio.items():
                                    if t == ticker:
                                        price = base_sim_results[ticker]['expected_price'][0]
                                        capital_adjustment -= details['quantity'] * price
                
                # Check in AI recommendations
                if 'ai_recommendations' in st.session_state:
                    for ai_action in st.session_state.ai_recommendations:
                        if ai_action["id"] == action_id:
                            if ai_action["type"] == "add_lumpsum_all" and "amount" in ai_action:
                                capital_adjustment += ai_action["amount"]
                            elif ai_action["type"] == "add_lumpsum_stock" and "amount" in ai_action:
                                capital_adjustment += ai_action["amount"]
                            elif ai_action["type"] == "book_profit" and "ticker" in ai_action and "percentage" in ai_action:
                                ticker = ai_action["ticker"]
                                percentage = ai_action["percentage"] / 100.0
                                for t, details in portfolio.items():
                                    if t == ticker:
                                        price = base_sim_results[ticker]['expected_price'][0]
                                        shares_sold = details['quantity'] * percentage
                                        capital_adjustment -= shares_sold * price
                            elif ai_action["type"] == "close_position" and "ticker" in ai_action:
                                ticker = ai_action["ticker"]
                                for t, details in portfolio.items():
                                    if t == ticker:
                                        price = base_sim_results[ticker]['expected_price'][0]
                                        capital_adjustment -= details['quantity'] * price
        
        # Get the expected return directly from the simulation results
        expected_return = current_results['portfolio_stats']['expected_return']
        
        # Adjust the initial value by adding/subtracting capital based on actions
        adjusted_initial_value = current_initial_value + capital_adjustment
        
        # Recalculate the adjusted return based on the adjusted initial value
        if adjusted_initial_value > 0:
            adjusted_return = ((current_final_value / adjusted_initial_value) - 1) * 100
        else:
            adjusted_return = expected_return  # Fallback to the simulation's calculation
        
        st.markdown("<h3>Updated Portfolio Metrics with Events & Actions</h3>", unsafe_allow_html=True)
        
        updated_cols = st.columns(4)
        
        with updated_cols[0]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Initial Value</div>
                <div class='metric-value'>{format_currency(adjusted_initial_value)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with updated_cols[1]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Expected Final Value</div>
                <div class='metric-value'>{format_currency(current_final_value)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with updated_cols[2]:
            return_class = "positive" if adjusted_return > 0 else "negative"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Expected Return</div>
                <div class='metric-value {return_class}'>{adjusted_return:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with updated_cols[3]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Portfolio Volatility</div>
                <div class='metric-value'>{current_volatility:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
    
    # Add AI-recommended actions section
    st.markdown("<h2>AI-Recommended Actions for Your Portfolio</h2>", unsafe_allow_html=True)
    st.write("")
    
    # Generate AI recommendations if not already in session state
    if 'ai_recommendations' not in st.session_state:
        with st.spinner("Analyzing your portfolio for personalized recommendations..."):
            st.session_state.ai_recommendations = get_ai_recommended_actions(portfolio, st.session_state.investment_personality)[:4]
            
            # Add manual recommendations for ITC and ZOMATO for consumer sector diversification
            # ITC recommendation
            st.session_state.ai_recommendations.append({
                "id": "add_itc_stock",
                "type": "add_lumpsum_stock",
                "title": "Add ITC to Portfolio",
                "description": "Diversify into consumer sector with ITC, a stable FMCG company with strong dividend yield",
                "icon": "🛒",
                "ticker": "ITC.NS",
                "amount": 25000,
                "month": 4
            })
            
            # ZOMATO recommendation
            st.session_state.ai_recommendations.append({
                "id": "add_zomato_stock",
                "type": "add_lumpsum_stock",
                "title": "Add ZOMATO to Portfolio",
                "description": "Diversify into consumer tech sector with ZOMATO, a growing food delivery platform",
                "icon": "🍔",
                "ticker": "ZOMATO.NS",
                "amount": 25000,
                "month": 5
            })
    
    # Display AI-recommended actions in a grid
    if st.session_state.ai_recommendations:
        st.markdown("<p>Our AI has analyzed your portfolio and investment personality to recommend these actions</p>", unsafe_allow_html=True)

        ai_action_cols = st.columns(3)
        
        for i, action in enumerate(st.session_state.ai_recommendations):
            with ai_action_cols[i % 3]:
                is_selected = action["id"] in st.session_state.selected_actions
                card_class = "card card-selected" if is_selected else "card"
                
                # Create clickable card
                action_type = ""
                badge_color = ""
                
                if "add_lumpsum" in action["type"]:
                    action_type = "Investment"
                    badge_color = "high"
                elif "book_profit" in action["type"]:
                    action_type = "Profit Booking"
                    badge_color = "medium"
                elif "close_position" in action["type"]:
                    action_type = "Exit"
                    badge_color = "low"
                
                # Define a callback function for when the card is clicked
                def toggle_ai_action(action_id=action["id"]):
                    if action_id in st.session_state.selected_actions:
                        st.session_state.selected_actions.remove(action_id)
                    else:
                        st.session_state.selected_actions.append(action_id)
                    # Set a flag to trigger rerun
                    st.session_state.trigger_rerun = True
                
                # Prepare badge HTML
                badge_html = f'<span class="badge badge-{badge_color}">{action_type}</span>'
                
                # Add stock-specific badges
                if "ticker" in action:
                    ticker = action["ticker"]
                    if "TCS" in ticker:
                        badge_html += ' <span class="badge badge-tech">TCS</span>'
                    elif "RELIANCE" in ticker:
                        badge_html += ' <span class="badge badge-energy">Reliance</span>'
                    elif "HDFC" in ticker or "SBIN" in ticker:
                        badge_html += ' <span class="badge badge-finance">Financial</span>'
                    elif "INFY" in ticker:
                        badge_html += ' <span class="badge badge-tech">Infosys</span>'
                
                # Create a container for the card
                with st.container():
                    # Add the card content
                    st.markdown(f"""
                    <div class='{card_class}' id='ai-action-{action["id"]}'>
                        <div class='card-title'>{action["icon"]} {action["title"]} <span style='float: right; font-size: 0.8em; color: #63B3ED;'>Month {action["month"]}</span></div>
                        <div class='card-description'>{action["description"]}</div>
                        <div style='margin-top: 10px;'>
                            {badge_html}
                            <div style='text-align: right; font-size: 0.7em; color: #63B3ED; margin-top: 5px;'>
                                {is_selected and "✓ Selected" or "Click to select"}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add an invisible button that covers the entire card
                    st.button("", key=f"btn_ai_action_{action['id']}", on_click=toggle_ai_action)
    else:
        st.info("No AI recommendations available at this time. Please try again later.")
    

    # Display events and actions as cards
    st.markdown("<h2>Select Recent Market Events to Simulate</h2>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<p>Click on events to see how they might affect your portfolio</p>", unsafe_allow_html=True)
    
    # Display events in a grid
    events = get_events()
    event_cols = st.columns(3)
    
    for i, event in enumerate(events):
        with event_cols[i % 3]:
            is_selected = event["id"] in st.session_state.selected_events
            card_class = "card card-selected" if is_selected else "card"
            
            # Create clickable card
            probability_class = f"badge badge-{event['probability'].lower()}"
            
            # Define a callback function for when the card is clicked
            def toggle_event(event_id=event["id"]):
                if event_id in st.session_state.selected_events:
                    st.session_state.selected_events.remove(event_id)
                else:
                    st.session_state.selected_events.append(event_id)
                # Set a flag to trigger rerun
                st.session_state.trigger_rerun = True
            
            # Create a container for the card
            with st.container():
                # Add the card content
                st.markdown(f"""
                <div class='{card_class}' id='event-{event["id"]}'>
                    <div class='card-title'>{event["icon"]} {event["title"]} <span style='float: right; font-size: 0.8em; color: #63B3ED;'>Month {event["month"]}</span></div>
                    <div class='card-description'>{event["description"]}</div>
                    <div style='margin-top: 10px;'>
                        <span class='{probability_class}'>{event["probability"]} Probability</span>
                        <div style='text-align: right; font-size: 0.7em; color: #63B3ED; margin-top: 5px;'>
                            {is_selected and "✓ Selected" or "Click to select"}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add an invisible button that covers the entire card
                st.markdown("""
                <style>
                .element-container:has(button) {
                    position: relative;
                    margin-top: -130px;  /* Adjust based on your card height */
                    z-index: 1;
                }
                
                .element-container:has(button) button {
                    width: 100%;
                    height: 130px;  /* Adjust based on your card height */
                    opacity: 0;
                    cursor: pointer;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Add the invisible button
                st.button("", key=f"btn_event_{event['id']}", on_click=toggle_event)
                

    # Add custom scenario section
    st.markdown("<h2>Your Custom Scenario</h2>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<p>Describe your custom market event or investment action to simulate</p>", unsafe_allow_html=True)
    
    # Initialize session state for custom scenarios
    if 'custom_scenarios' not in st.session_state:
        st.session_state.custom_scenarios = []
    
    # Text input for custom scenario
    custom_scenario_description = st.markdown("""
        <textarea
        id="customScenarioDescription"
        name="customScenarioDescription"
        rows="5"
        placeholder="Example: 'A global chip shortage affects technology stocks in month 4' or 'I invest ₹200,000 in HDFC Bank in month 6'"
        style="width: 100%; padding: 0.75rem; font-size: 1rem; border-radius: 0.5rem; border: 1px solid #ccc; resize: vertical; margin-top: 0.5rem; margin-bottom: 1rem;"
        ></textarea>
    """, unsafe_allow_html=True)


    st.write("")  # Add space between textarea and button
    custom_generate_button = st.button("Generate Custom Scenario", key="generate_custom_scenario_btn")

    # Keep the custom styling but remove position/z-index
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        background-color: #4299E1 !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        display: block !important;
        height: auto !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        margin-top: 1rem !important;
    }

    div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
        background-color: #3182CE !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    
    if custom_generate_button:
        if custom_scenario_description:
            with st.spinner("Generating custom scenario..."):
                scenario, scenario_type = generate_custom_scenario(custom_scenario_description, portfolio)
                
                if scenario:
                    # Add to session state
                    st.session_state.custom_scenarios.append({
                        "scenario": scenario,
                        "type": scenario_type
                    })
                    
                    # Add to selected events or actions
                    if scenario_type == "event":
                        st.session_state.selected_events.append(scenario["id"])
                    else:
                        st.session_state.selected_actions.append(scenario["id"])
                    
                    # Set flag to trigger rerun
                    st.session_state.trigger_rerun = True
                else:
                    st.error(f"Failed to generate scenario: {scenario_type}")
    
    # Display custom scenarios
    if st.session_state.custom_scenarios:
        st.markdown("<h3>Your Custom Scenarios</h3>", unsafe_allow_html=True)
        
        # Create columns for custom scenario cards
        custom_cols = st.columns(2)
        
        for i, custom_item in enumerate(st.session_state.custom_scenarios):
            scenario = custom_item["scenario"]
            scenario_type = custom_item["type"]
            
            with custom_cols[i % 2]:
                # Determine if selected
                is_selected = False
                if scenario_type == "event":
                    is_selected = scenario["id"] in st.session_state.selected_events
                else:
                    is_selected = scenario["id"] in st.session_state.selected_actions
                
                card_class = "card card-selected" if is_selected else "card"
                
                # Define badge based on type
                if scenario_type == "event":
                    probability = scenario.get("probability", "Medium")
                    badge_html = f'<span class="badge badge-{probability.lower()}">{probability} Probability</span>'
                else:
                    action_type = ""
                    badge_color = ""
                    
                    if "add_lumpsum" in scenario["type"]:
                        action_type = "Investment"
                        badge_color = "high"
                    elif "book_profit" in scenario["type"]:
                        action_type = "Profit Booking"
                        badge_color = "medium"
                    elif "close_position" in scenario["type"]:
                        action_type = "Exit"
                        badge_color = "low"
                    
                    badge_html = f'<span class="badge badge-{badge_color}">{action_type}</span>'
                    
                    # Add stock-specific badges
                    if "ticker" in scenario:
                        ticker = scenario["ticker"]
                        sector = ""
                        if "TCS" in ticker:
                            badge_html += ' <span class="badge badge-tech">TCS</span>'
                        elif "RELIANCE" in ticker:
                            badge_html += ' <span class="badge badge-energy">Reliance</span>'
                        elif "HDFC" in ticker or "SBIN" in ticker:
                            badge_html += ' <span class="badge badge-finance">Financial</span>'
                
                # Define callback for toggling
                def toggle_custom_scenario(scenario_id=scenario["id"], s_type=scenario_type):
                    if s_type == "event":
                        if scenario_id in st.session_state.selected_events:
                            st.session_state.selected_events.remove(scenario_id)
                        else:
                            st.session_state.selected_events.append(scenario_id)
                    else:
                        if scenario_id in st.session_state.selected_actions:
                            st.session_state.selected_actions.remove(scenario_id)
                        else:
                            st.session_state.selected_actions.append(scenario_id)
                    
                    # Set flag to trigger rerun
                    st.session_state.trigger_rerun = True
                
                # Create a container for the card
                with st.container():
                    # Add the card content
                    st.markdown(f"""
                    <div class='{card_class}' id='custom-{scenario["id"]}'>
                        <div class='card-title'>{scenario["icon"]} {scenario["title"]} <span style='float: right; font-size: 0.8em; color: #63B3ED;'>Month {scenario["month"]}</span></div>
                        <div class='card-description'>{scenario["description"]}</div>
                        <div style='margin-top: 10px;'>
                            {badge_html}
                            <div style='text-align: right; font-size: 0.7em; color: #63B3ED; margin-top: 5px;'>
                                {is_selected and "✓ Selected" or "Click to select"}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add an invisible button that covers the entire card with a unique key using both scenario ID and index
                    st.button("", key=f"btn_custom_{scenario['id']}_{i}", on_click=toggle_custom_scenario)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #666;'>
        <p>This simulation is for educational purposes only and does not constitute investment advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we need to rerun the app
    if 'trigger_rerun' in st.session_state and st.session_state.trigger_rerun:
        st.session_state.trigger_rerun = False
        st.rerun()

if __name__ == "__main__":
    # Initialize the trigger_rerun flag if it doesn't exist
    if 'trigger_rerun' not in st.session_state:
        st.session_state.trigger_rerun = False
    
    main()
