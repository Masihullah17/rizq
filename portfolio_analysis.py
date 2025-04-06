import os
import json
from llm_api import chat_with_deepseek

def get_stock_summary(ticker):
    """
    Get the summary for a specific stock from the data/summaries folder.
    
    Args:
        ticker (str): The stock ticker symbol (without .NS extension)
        
    Returns:
        str: The stock summary text or empty string if not found
    """
    # Remove .NS extension if present
    ticker = ticker.replace(".NS", "")
    
    # Try to read the summary file
    try:
        with open(f"data/summaries/{ticker}_summary.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Summary not found for {ticker}")
        return ""

def analyze_portfolio_with_llm(portfolio, investment_personality):
    """
    Analyze the portfolio using the DeepSeek reasoning model and generate
    top 5 actions to improve portfolio performance.
    
    Args:
        portfolio (dict): The user's portfolio data
        investment_personality (str): The user's investment personality analysis
        
    Returns:
        list: List of recommended actions in the format compatible with the app
    """
    # Get summaries for each stock in the portfolio
    stock_summaries = {}
    for ticker in portfolio.keys():
        # Extract the base ticker without .NS
        base_ticker = ticker.replace(".NS", "")
        summary = get_stock_summary(base_ticker)
        if summary:
            stock_summaries[ticker] = summary
    
    # Create a system prompt for the DeepSeek model
    system_prompt = """
    You are a professional financial advisor specializing in portfolio optimization. Your task is to analyze a user's investment portfolio and provide the top 5 specific actions they should take to improve their portfolio's performance.

    Based on the user's investment personality, current portfolio composition, and detailed stock summaries, recommend 5 specific actions from the following categories:
    1. BOOK_PROFIT: Recommend selling a specific percentage of a stock that has reached its target or is overvalued
    2. CLOSE_POSITION: Recommend completely exiting a position due to fundamental concerns or better alternatives
    3. ADD_LUMPSUM_STOCK: Recommend adding a specific amount to an existing stock with strong growth potential
    4. ADD_LUMPSUM_ALL: Recommend adding a specific amount distributed across all stocks or a subset of stocks
    5. REPLACE_STOCK: Recommend replacing one stock with another specific stock (from the available summaries)
    6. SECTOR_REBALANCE: Recommend reducing exposure to one sector and increasing exposure to another

    For each recommendation, provide:
    1. The specific action type (from the categories above)
    2. The specific stock ticker(s) involved
    3. The specific amount or percentage involved
    4. A clear, concise rationale (1-2 sentences)
    5. A suggested timeline (immediate, within 1 month, within 3 months)
    6. An icon emoji that represents the action

    Your response should be in JSON format with an array of 5 action objects, each containing:
    {
        "id": "unique_id",
        "type": "action_type", (must be one of: add_lumpsum_all, add_lumpsum_stock, book_profit, close_position)
        "title": "Action Title",
        "description": "Detailed description of the action",
        "icon": "emoji",
        "month": number (1-12),
        ... additional fields based on action type:
        - For add_lumpsum_all: "amount": number
        - For add_lumpsum_stock: "ticker": "stock_ticker", "amount": number
        - For book_profit: "ticker": "stock_ticker", "percentage": number
        - For close_position: "ticker": "stock_ticker"
    }
    """
    
    # Create a user prompt with the portfolio information and stock summaries
    user_prompt = f"""
    Please analyze this investment portfolio and provide the top 5 actions to improve performance:
    
    ## Investment Personality
    {investment_personality}
    
    ## Current Portfolio
    {json.dumps(portfolio, indent=2)}
    
    ## Stock Summaries
    """
    
    # Add each stock summary to the prompt
    for ticker, summary in stock_summaries.items():
        user_prompt += f"\n### {ticker}\n{summary}\n"
    
    # Add instructions for the response format
    user_prompt += """
    Based on this information, please provide the top 5 recommended actions to improve the portfolio's performance.
    Return ONLY the JSON array of 5 action objects without any additional text or explanation.
    """
    
    try:
        # Call the DeepSeek model to generate the recommendations
        response = chat_with_deepseek(user_prompt, model="deepseek-r1", system_prompt=system_prompt)
        
        # Parse the JSON response
        # First, try to find JSON within the response if it's not a pure JSON response
        import re
        json_match = re.search(r'(\[[\s\S]*\])', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
            
        # Parse the JSON
        try:
            recommendations = json.loads(json_str)
            
            # Validate and format the recommendations
            formatted_recommendations = []
            for rec in recommendations:
                # Ensure required fields are present
                if all(field in rec for field in ["id", "type", "title", "description", "icon", "month"]):
                    # Add additional validation based on action type
                    if rec["type"] == "add_lumpsum_all" and "amount" in rec:
                        formatted_recommendations.append(rec)
                    elif rec["type"] == "add_lumpsum_stock" and "ticker" in rec and "amount" in rec:
                        formatted_recommendations.append(rec)
                    elif rec["type"] == "book_profit" and "ticker" in rec and "percentage" in rec:
                        formatted_recommendations.append(rec)
                    elif rec["type"] == "close_position" and "ticker" in rec:
                        formatted_recommendations.append(rec)
            
            return formatted_recommendations
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return []
    
    except Exception as e:
        print(f"Error generating portfolio recommendations: {str(e)}")
        return []

def get_ai_recommended_actions(portfolio, investment_personality):
    """
    Get AI-recommended actions for the portfolio.
    This function is a wrapper around analyze_portfolio_with_llm that handles caching and formatting.
    
    Args:
        portfolio (dict): The user's portfolio data
        investment_personality (str): The user's investment personality analysis
        
    Returns:
        list: List of recommended actions in the format compatible with the app
    """
    # Generate recommendations
    recommendations = analyze_portfolio_with_llm(portfolio, investment_personality)
    
    # If no recommendations were generated, return an empty list
    if not recommendations:
        return []
    
    import random
    months = [1, 3, 5, 7, 9]
    random.shuffle(months)
    for i in range(len(recommendations)):
        recommendations[i]['month'] = months[i]
    
    # Return the recommendations
    return recommendations
