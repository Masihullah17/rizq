# RIZQ - Hyper-personalized Wealth Assistant

RIZQ is a hyper-personalized wealth assistant designed to help retail investors make more informed investment decisions. The application combines Monte Carlo simulations, market event modeling, and AI-driven insights to provide personalized portfolio analysis and recommendations.

![RIZQ](snapshot.png)

## 🚀 Features

### Portfolio Monte Carlo Simulation
- Project future portfolio performance over a one-year horizon
- Visualize expected returns, volatility, and confidence intervals
- Compare performance against market benchmarks
- View historical portfolio performance (past 6 months) alongside future projections

### Market Event Simulation
- Test portfolio resilience against various market scenarios (corrections, sector booms/busts, etc.)
- Visualize the impact of market events on portfolio performance
- Understand sector-specific vulnerabilities

### Investment Action Planning
- Simulate the impact of potential investment actions (adding funds, booking profits, etc.)
- Compare different investment strategies
- Optimize timing of investment actions

### AI-Powered Insights
- Investment personality analysis based on portfolio composition
- Personalized investment recommendations
- Custom scenario generation from natural language descriptions

### Interactive Visualization
- Animated portfolio performance charts
- Interactive selection of events and actions
- Visual comparison of different scenarios

## 🛠️ Technologies Used

- **Python 3.9+**: Primary development language
- **Streamlit**: Web interface and interactive components
- **NumPy/Pandas**: Data processing and analysis
- **Plotly**: Interactive data visualization
- **OpenAI API (GPT-4o)**: Natural language processing and AI-driven insights
- **DeepSeek API**: Alternative AI model for investment analysis

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key
- DeepSeek API key

## 🔧 Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rizq.git
   cd rizq
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

4. Run the application:
   ```bash
   streamlit run portfolio_streamlit_app.py
   ```