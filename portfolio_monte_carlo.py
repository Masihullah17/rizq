import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
import yfinance as yf
from typing import Dict, List, Tuple, Any, Optional, Union
import json

class PortfolioMonteCarloSimulator:
    """
    A comprehensive Monte Carlo simulator for stock portfolios that incorporates
    various market events and portfolio actions over a simulation period.
    """
    
    # Define event impacts by type and sector
    EVENT_IMPACTS = {
        # Market-wide events
        "market_crash": {
            "duration": (10, 30),  # (min_days, max_days)
            "base_drift": -0.15,   # Base drift effect
            "vol_multiplier": 2.5, # Volatility increase
            "sector_impacts": {    # Sector-specific modifiers
                "Technology": 1.2,  # Tech gets hit harder in crashes
                "Healthcare": 0.7,  # Healthcare more resilient
                "Financial": 1.3,   # Financial hit harder
                "Energy": 0.9,
                "Consumer": 0.8,
                "Industrial": 1.1,
                "Utilities": 0.6,   # Utilities most resilient
                "Materials": 1.0,
                "Real Estate": 1.2,
                "Communication": 1.0,
                "default": 1.0      # Default multiplier for unlisted sectors
            }
        },
        "market_correction": {
            "duration": (5, 15),
            "base_drift": -0.08,
            "vol_multiplier": 1.8,
            "sector_impacts": {
                "Technology": 1.3,
                "Healthcare": 0.8,
                "Financial": 1.1,
                "Energy": 0.9,
                "Consumer": 0.9,
                "Industrial": 1.0,
                "Utilities": 0.7,
                "Materials": 1.0,
                "Real Estate": 1.1,
                "Communication": 1.1,
                "default": 1.0
            }
        },
        "bull_run": {
            "duration": (20, 60),
            "base_drift": 0.12,
            "vol_multiplier": 1.2,
            "sector_impacts": {
                "Technology": 1.4,
                "Healthcare": 0.9,
                "Financial": 1.2,
                "Energy": 0.8,
                "Consumer": 1.1,
                "Industrial": 1.1,
                "Utilities": 0.6,
                "Materials": 1.0,
                "Real Estate": 1.0,
                "Communication": 1.3,
                "default": 1.0
            }
        },
        "bear_run": {
            "duration": (20, 45),
            "base_drift": -0.10,
            "vol_multiplier": 1.7,
            "sector_impacts": {
                "Technology": 1.3,
                "Healthcare": 0.8,
                "Financial": 1.2,
                "Energy": 1.0,
                "Consumer": 0.9,
                "Industrial": 1.1,
                "Utilities": 0.7,
                "Materials": 1.0,
                "Real Estate": 1.2,
                "Communication": 1.1,
                "default": 1.0
            }
        },
        "economic_downturn": {
            "duration": (30, 90),
            "base_drift": -0.08,
            "vol_multiplier": 1.5,
            "sector_impacts": {
                "Technology": 1.1,
                "Healthcare": 0.7,
                "Financial": 1.4,
                "Energy": 1.2,
                "Consumer": 1.3,
                "Industrial": 1.3,
                "Utilities": 0.6,
                "Materials": 1.2,
                "Real Estate": 1.4,
                "Communication": 1.0,
                "default": 1.0
            }
        },
        
        # Policy events
        "us_tariff_increase": {
            "duration": (10, 30),
            "base_drift": -0.05,
            "vol_multiplier": 1.3,
            "sector_impacts": {
                "Technology": 1.2,
                "Healthcare": 0.7,
                "Financial": 0.8,
                "Energy": 0.9,
                "Consumer": 1.3,  # Consumer goods affected by tariffs
                "Industrial": 1.4, # Manufacturing affected
                "Utilities": 0.5,
                "Materials": 1.2,
                "Real Estate": 0.7,
                "Communication": 0.8,
                "default": 1.0
            }
        },
        "us_tariff_decrease": {
            "duration": (10, 20),
            "base_drift": 0.04,
            "vol_multiplier": 0.9,
            "sector_impacts": {
                "Technology": 1.1,
                "Healthcare": 0.8,
                "Financial": 0.9,
                "Energy": 0.8,
                "Consumer": 1.3,  # Consumer goods benefit
                "Industrial": 1.4, # Manufacturing benefits
                "Utilities": 0.6,
                "Materials": 1.1,
                "Real Estate": 0.7,
                "Communication": 0.8,
                "default": 1.0
            }
        },
        "rbi_rate_cut": {
            "duration": (5, 15),
            "base_drift": 0.06,
            "vol_multiplier": 0.8,
            "sector_impacts": {
                "Technology": 1.0,
                "Healthcare": 0.8,
                "Financial": 1.5,  # Financial sector most affected by rate cuts
                "Energy": 0.9,
                "Consumer": 1.1,
                "Industrial": 1.0,
                "Utilities": 1.2,
                "Materials": 0.9,
                "Real Estate": 1.4,  # Real estate benefits from lower rates
                "Communication": 0.9,
                "default": 1.0
            }
        },
        "rbi_rate_hike": {
            "duration": (5, 15),
            "base_drift": -0.05,
            "vol_multiplier": 1.2,
            "sector_impacts": {
                "Technology": 1.1,
                "Healthcare": 0.7,
                "Financial": 1.4,  # Financial sector most affected by rate hikes
                "Energy": 0.8,
                "Consumer": 1.0,
                "Industrial": 1.0,
                "Utilities": 1.1,
                "Materials": 0.9,
                "Real Estate": 1.5,  # Real estate hurt by higher rates
                "Communication": 0.9,
                "default": 1.0
            }
        },
        
        # Sector-specific events
        "tech_sector_boom": {
            "duration": (15, 45),
            "base_drift": 0.10,
            "vol_multiplier": 1.4,
            "sector_impacts": {
                "Technology": 2.0,  # Tech stocks benefit greatly
                "Healthcare": 0.5,
                "Financial": 0.7,
                "Energy": 0.4,
                "Consumer": 0.6,
                "Industrial": 0.5,
                "Utilities": 0.3,
                "Materials": 0.4,
                "Real Estate": 0.5,
                "Communication": 1.5,  # Communication also benefits
                "default": 0.5
            }
        },
        "tech_sector_crash": {
            "duration": (10, 30),
            "base_drift": -0.12,
            "vol_multiplier": 1.8,
            "sector_impacts": {
                "Technology": 2.0,  # Tech stocks hit hardest
                "Healthcare": 0.5,
                "Financial": 0.8,
                "Energy": 0.4,
                "Consumer": 0.6,
                "Industrial": 0.5,
                "Utilities": 0.3,
                "Materials": 0.4,
                "Real Estate": 0.5,
                "Communication": 1.5,  # Communication also hit
                "default": 0.5
            }
        },
        "energy_sector_boom": {
            "duration": (15, 45),
            "base_drift": 0.08,
            "vol_multiplier": 1.3,
            "sector_impacts": {
                "Technology": 0.5,
                "Healthcare": 0.5,
                "Financial": 0.7,
                "Energy": 2.0,  # Energy stocks benefit greatly
                "Consumer": 0.7,
                "Industrial": 0.9,
                "Utilities": 1.2,  # Utilities also benefit
                "Materials": 1.1,
                "Real Estate": 0.5,
                "Communication": 0.5,
                "default": 0.6
            }
        },
        "energy_sector_crash": {
            "duration": (10, 30),
            "base_drift": -0.10,
            "vol_multiplier": 1.7,
            "sector_impacts": {
                "Technology": 0.5,
                "Healthcare": 0.5,
                "Financial": 0.7,
                "Energy": 2.0,  # Energy stocks hit hardest
                "Consumer": 0.7,
                "Industrial": 0.9,
                "Utilities": 1.2,  # Utilities also affected
                "Materials": 1.1,
                "Real Estate": 0.5,
                "Communication": 0.5,
                "default": 0.6
            }
        },
        "financial_sector_crisis": {
            "duration": (15, 45),
            "base_drift": -0.14,
            "vol_multiplier": 2.0,
            "sector_impacts": {
                "Technology": 0.8,
                "Healthcare": 0.6,
                "Financial": 2.0,  # Financial stocks hit hardest
                "Energy": 0.7,
                "Consumer": 0.9,
                "Industrial": 0.8,
                "Utilities": 0.6,
                "Materials": 0.7,
                "Real Estate": 1.5,  # Real estate also hit due to financing issues
                "Communication": 0.7,
                "default": 0.8
            }
        }
    }
    
    def __init__(self, 
                 portfolio: Dict[str, Dict[str, Any]], 
                 forecast_days: int = 252,
                 num_simulations: int = 1000,
                 seed: int = 42,
                 historical_days: int = 756):  # ~3 years of data
        """
        Initialize the Monte Carlo simulator with portfolio data.
        
        Args:
            portfolio: Dictionary with stock details including:
                      - ticker: Stock symbol
                      - sector: Stock sector
                      - quantity: Number of shares
                      - buy_price: Purchase price per share
                      - current_price: Current price per share
                      - analyst_targets: Dict with min, avg, max price targets
            forecast_days: Number of trading days to simulate (default: 252 = 1 year)
            num_simulations: Number of Monte Carlo simulations to run
            seed: Random seed for reproducibility
            historical_days: Number of historical days to use for volatility calculation
        """
        self.portfolio = portfolio
        self.forecast_days = forecast_days
        self.num_simulations = num_simulations
        self.seed = seed
        self.historical_days = historical_days
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize simulation results
        self.simulation_results = {}
        self.portfolio_values = None
        self.event_log = []
        self.action_log = []
        
        # Track portfolio changes over time
        self.portfolio_history = {0: self._get_initial_portfolio_snapshot()}
    
    def _get_initial_portfolio_snapshot(self) -> Dict[str, Any]:
        """Create an initial snapshot of the portfolio."""
        total_value = 0
        stocks = {}
        
        for ticker, details in self.portfolio.items():
            stock_value = details['quantity'] * details['current_price']
            total_value += stock_value
            stocks[ticker] = {
                'quantity': details['quantity'],
                'price': details['current_price'],
                'value': stock_value
            }
        
        return {
            'total_value': total_value,
            'stocks': stocks,
            'timestamp': 0  # Day 0
        }
    
    def _generate_random_events(self, events_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate random events based on the provided events list.
        
        Args:
            events_list: List of event dictionaries with event type and optional timing
            
        Returns:
            List of processed event dictionaries with timing information
        """
        processed_events = []
        occupied_days = set()
        
        for event in events_list:
            event_type = event['type']
            
            # Skip if event type not in our defined impacts
            if event_type not in self.EVENT_IMPACTS:
                continue
                
            impact = self.EVENT_IMPACTS[event_type]
            
            # Use specified day or generate random day
            if 'day' in event:
                start_day = event['day']
            elif 'month' in event:
                # Convert month to approximate day (month * ~21 trading days)
                start_day = min((event['month'] - 1) * 21, self.forecast_days - 1)
            else:
                # Random start day ensuring event fits within forecast period
                min_duration = impact['duration'][0]
                max_start = self.forecast_days - min_duration - 1
                if max_start <= 0:
                    continue
                
                # Try to find a non-overlapping period
                found_slot = False
                for _ in range(10):  # Try 10 times to find a non-overlapping slot
                    candidate_start = random.randint(0, max_start)
                    duration = random.randint(impact['duration'][0], impact['duration'][1])
                    candidate_end = candidate_start + duration
                    
                    # Check if this period overlaps with any existing events
                    if not any(day in occupied_days for day in range(candidate_start, candidate_end + 1)):
                        start_day = candidate_start
                        found_slot = True
                        break
                
                if not found_slot:
                    # If we couldn't find a non-overlapping slot, just pick a random day
                    start_day = random.randint(0, max_start)
            
            # Calculate duration and end day
            if 'duration' in event:
                duration = event['duration']
            else:
                duration = random.randint(impact['duration'][0], impact['duration'][1])
            
            end_day = min(start_day + duration, self.forecast_days - 1)
            
            # Mark these days as occupied
            occupied_days.update(range(start_day, end_day + 1))
            
            # Add to processed events
            processed_events.append({
                'type': event_type,
                'start_day': start_day,
                'end_day': end_day,
                'base_drift': impact['base_drift'],
                'vol_multiplier': impact['vol_multiplier'],
                'sector_impacts': impact['sector_impacts'],
                'description': event.get('description', f"{event_type} event")
            })
            
            # Add to event log
            self.event_log.append({
                'day': start_day,
                'type': event_type,
                'duration': duration,
                'description': event.get('description', f"{event_type} event")
            })
        
        return processed_events
    
    def _process_actions(self, actions_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process portfolio actions and schedule them.
        
        Args:
            actions_list: List of action dictionaries
            
        Returns:
            List of processed action dictionaries with timing information
        """
        processed_actions = []
        
        for action in actions_list:
            action_type = action['type']
            
            # Use specified day or generate random day
            if 'day' in action:
                day = action['day']
            elif 'month' in action:
                # Convert month to approximate day (month * ~21 trading days)
                day = min((action['month'] - 1) * 21, self.forecast_days - 1)
            else:
                # Random day
                day = random.randint(0, self.forecast_days - 1)
            
            # Add day to action
            action_copy = action.copy()
            action_copy['day'] = day
            
            # Add to processed actions
            processed_actions.append(action_copy)
            
            # Add to action log
            self.action_log.append({
                'day': day,
                'type': action_type,
                'details': action_copy
            })
        
        # Sort actions by day
        processed_actions.sort(key=lambda x: x['day'])
        
        return processed_actions
    
    def _apply_action_to_portfolio(self, day: int, action: Dict[str, Any], 
                                  current_portfolio: Dict[str, Dict[str, Any]], 
                                  simulated_prices: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Apply an action to the portfolio at a specific day.
        
        Args:
            day: The simulation day
            action: Action dictionary
            current_portfolio: Current portfolio state
            simulated_prices: Dictionary of simulated prices for each stock
            
        Returns:
            Updated portfolio dictionary
        """
        action_type = action['type']
        updated_portfolio = {ticker: details.copy() for ticker, details in current_portfolio.items()}
        
        if action_type == 'add_lumpsum_all':
            # Add lumpsum amount distributed proportionally across all stocks
            total_amount = action['amount']
            
            # Use the first simulation for portfolio adjustments
            # This ensures we're working with scalar values, not arrays
            total_value = sum(details['quantity'] * float(simulated_prices[ticker][day, 0]) 
                             for ticker, details in current_portfolio.items())
            
            for ticker, details in current_portfolio.items():
                if total_value > 0:
                    stock_value = details['quantity'] * float(simulated_prices[ticker][day, 0])
                    proportion = stock_value / total_value
                    additional_amount = total_amount * proportion
                    additional_shares = additional_amount / float(simulated_prices[ticker][day, 0])
                    # Floor the quantity to ensure whole shares only
                    updated_portfolio[ticker]['quantity'] = np.floor(updated_portfolio[ticker]['quantity'] + additional_shares)
            
        elif action_type == 'add_lumpsum_stock':
            # Add lumpsum to specific stock
            ticker = action['ticker']
            amount = action['amount']
            
            if ticker in updated_portfolio:
                # Use the first simulation for consistency
                price = float(simulated_prices[ticker][day, 0])
                additional_shares = amount / price
                # Floor the quantity to ensure whole shares only
                updated_portfolio[ticker]['quantity'] = np.floor(updated_portfolio[ticker]['quantity'] + additional_shares)
        
        elif action_type == 'increase_sip':
            # Increase regular investment amount
            sip_increase = action['percentage'] / 100.0
            
            for ticker, details in current_portfolio.items():
                if 'sip_amount' in details:
                    details['sip_amount'] *= (1 + sip_increase)
        
        elif action_type == 'book_profit':
            # Book profit by selling a percentage of a stock
            ticker = action['ticker']
            percentage = action['percentage'] / 100.0
            
            if ticker in updated_portfolio:
                shares_to_sell = np.floor(updated_portfolio[ticker]['quantity'] * percentage)
                updated_portfolio[ticker]['quantity'] -= shares_to_sell
        
        elif action_type == 'close_position':
            # Close position completely
            ticker = action['ticker']
            if ticker in updated_portfolio:
                updated_portfolio[ticker]['quantity'] = 0
        
        # Create a portfolio snapshot after this action
        self.portfolio_history[day] = self._create_portfolio_snapshot(day, updated_portfolio, simulated_prices)
        
        return updated_portfolio
    
    def _create_portfolio_snapshot(self, day: int, portfolio: Dict[str, Dict[str, Any]], 
                                  simulated_prices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create a snapshot of the portfolio at a specific day."""
        total_value = 0
        stocks = {}
        
        for ticker, details in portfolio.items():
            if details['quantity'] > 0:  # Only include stocks with non-zero quantity
                # Use the first simulation for consistency
                if isinstance(simulated_prices[ticker][day], np.ndarray):
                    price = float(simulated_prices[ticker][day, 0])
                else:
                    price = float(simulated_prices[ticker][day])
                    
                stock_value = details['quantity'] * price
                total_value += stock_value
                stocks[ticker] = {
                    'quantity': details['quantity'],
                    'price': price,
                    'value': stock_value
                }
        
        return {
            'total_value': total_value,
            'stocks': stocks,
            'timestamp': day
        }
    
    def _simulate_stock(self, ticker: str, details: Dict[str, Any], events: List[Dict[str, Any]]) -> np.ndarray:
        """
        Run Monte Carlo simulation for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            details: Stock details including sector, current price, etc.
            events: List of event dictionaries
            
        Returns:
            Array of simulated prices [days, simulations]
        """
        try:
            # Get historical data
            data = yf.download(ticker, period=f"{self.historical_days}d")['Close'].dropna()
            
            # If no data is available, use the current price as a constant
            if len(data) < 20:  # Need at least 20 days for meaningful statistics
                print(f"Warning: Insufficient historical data for {ticker}. Using current price.")
                simulated_prices = np.ones((self.forecast_days, self.num_simulations)) * details['current_price']
                return simulated_prices
            
            # Calculate log returns
            log_returns = np.log(1 + data.pct_change().dropna())
            
            # Calculate drift and volatility
            mu = float(log_returns.mean().iloc[0] if hasattr(log_returns.mean(), 'iloc') else log_returns.mean())
            sigma = float(log_returns.std().iloc[0] if hasattr(log_returns.std(), 'iloc') else log_returns.std())
            
            # Use exponentially weighted volatility for more recent emphasis
            ewm_std = log_returns.ewm(span=21).std()
            if hasattr(ewm_std, 'iloc'):
                if hasattr(ewm_std.iloc[-1], 'iloc'):
                    sigma = ewm_std.iloc[-1].iloc[0]
                else:
                    sigma = ewm_std.iloc[-1]
            else:
                sigma = ewm_std
            
            # Use analyst targets if available, but with very conservative estimates
            if 'analyst_targets' in details and details['analyst_targets']:
                avg_target = details['analyst_targets']['avg']
                # Calculate a realistic daily drift based on analyst targets
                # Limit annual return to a maximum of 30% for realism
                annual_return = min((avg_target / details['current_price']) - 1, 0.30)
                # Convert annual return to daily drift (very conservative)
                implied_drift = np.log(1 + annual_return) / self.forecast_days
                print(f"Using analyst target for {ticker}: {avg_target} (capped annual return: {annual_return:.2%}, daily drift: {implied_drift:.6f})")
            else:
                # Use historical drift but cap it to very conservative values
                implied_drift = min(max(mu - 0.5 * sigma ** 2, -0.0005), 0.0005)  # Cap between -0.05% and 0.05% daily
                print(f"No analyst target for {ticker}. Using capped historical drift: {implied_drift:.6f}")
            
            # Initialize price array
            simulated_prices = np.zeros((self.forecast_days, self.num_simulations))
            
            # Run simulations
            for sim in range(self.num_simulations):
                prices = [details['current_price']]
                
                for day in range(self.forecast_days):
                    # Default drift and volatility
                    drift = implied_drift
                    vol = sigma
                    
                    # Check if any events are active on this day
                    for event in events:
                        if event['start_day'] <= day <= event['end_day']:
                            # Get sector-specific impact
                            sector = details.get('sector', 'default')
                            sector_impact = event['sector_impacts'].get(sector, event['sector_impacts']['default'])
                            
                            # Apply event effects with some randomness
                            drift_effect = event['base_drift'] * sector_impact
                            drift += drift_effect * np.random.uniform(0.8, 1.2)  # Add some randomness
                            vol *= event['vol_multiplier'] * np.random.uniform(0.9, 1.1)  # Add some randomness
                    
                    # Generate random shock
                    shock = np.random.normal(loc=drift, scale=vol)
                    
                    # Calculate new price with a daily cap to prevent extreme movements
                    # Limit daily price changes to a maximum of 5% up or down
                    capped_shock = np.clip(shock, -0.05, 0.05)
                    new_price = prices[-1] * np.exp(capped_shock)
                    
                    # Add a sanity check to prevent unrealistic prices
                    # Limit the price to a reasonable range based on analyst targets or historical data
                    if 'analyst_targets' in details and details['analyst_targets']:
                        # Use analyst targets to set min and max bounds
                        min_target = details['analyst_targets'].get('min', details['current_price'] * 0.8)
                        max_target = details['analyst_targets'].get('max', details['current_price'] * 1.3)
                        # Ensure price stays within a reasonable range
                        new_price = max(min(new_price, max_target), min_target * 0.8)
                    else:
                        # Without analyst targets, use a more conservative range
                        min_price = details['current_price'] * 0.7
                        max_price = details['current_price'] * 1.3
                        new_price = max(min(new_price, max_price), min_price)
                    
                    prices.append(new_price)
                
                # Store simulation results
                simulated_prices[:, sim] = prices[1:]
            
            return simulated_prices
            
        except Exception as e:
            print(f"Error simulating {ticker}: {str(e)}")
            # Return constant price if simulation fails
            return np.ones((self.forecast_days, self.num_simulations)) * details['current_price']
    
    def run_simulation(self, events: List[Dict[str, Any]], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for the entire portfolio with events and actions.
        
        Args:
            events: List of event dictionaries
            actions: List of action dictionaries
            
        Returns:
            Dictionary with simulation results
        """
        # Process events and actions
        processed_events = self._generate_random_events(events)
        processed_actions = self._process_actions(actions)
        
        # Simulate each stock
        simulated_prices = {}
        for ticker, details in self.portfolio.items():
            print(f"Simulating {ticker}...")
            simulated_prices[ticker] = self._simulate_stock(ticker, details, processed_events)
            self.simulation_results[ticker] = {
                'prices': simulated_prices[ticker],
                'expected_price': np.mean(simulated_prices[ticker], axis=1),
                'min_price': np.min(simulated_prices[ticker], axis=1),
                'max_price': np.max(simulated_prices[ticker], axis=1),
                'percentile_05': np.percentile(simulated_prices[ticker], 5, axis=1),
                'percentile_95': np.percentile(simulated_prices[ticker], 95, axis=1)
            }
        
        # Initialize portfolio values array
        self.portfolio_values = np.zeros((self.forecast_days, self.num_simulations))
        
        # Apply actions and calculate portfolio value over time
        current_portfolio = {ticker: {'quantity': details['quantity']} 
                            for ticker, details in self.portfolio.items()}
        
        # Track actions by day
        actions_by_day = {}
        for action in processed_actions:
            day = action['day']
            if day not in actions_by_day:
                actions_by_day[day] = []
            actions_by_day[day].append(action)
        
        # Calculate portfolio value for each day and simulation
        for day in range(self.forecast_days):
            # Apply any actions scheduled for this day
            if day in actions_by_day:
                for action in actions_by_day[day]:
                    current_portfolio = self._apply_action_to_portfolio(
                        day, action, current_portfolio, simulated_prices)
            
            # Calculate portfolio value for this day across all simulations
            for sim in range(self.num_simulations):
                day_value = 0
                for ticker, details in current_portfolio.items():
                    if details['quantity'] > 0:  # Only include stocks with non-zero quantity
                        day_value += details['quantity'] * simulated_prices[ticker][day, sim]
                self.portfolio_values[day, sim] = day_value
            
            # Create portfolio snapshot if not already created by an action
            if day not in self.portfolio_history:
                self.portfolio_history[day] = self._create_portfolio_snapshot(
                    day, current_portfolio, {t: s[:, 0] for t, s in simulated_prices.items()})
        
        # Calculate portfolio statistics
        portfolio_stats = {
            'expected_value': np.mean(self.portfolio_values, axis=1),
            'min_value': np.min(self.portfolio_values, axis=1),
            'max_value': np.max(self.portfolio_values, axis=1),
            'percentile_05': np.percentile(self.portfolio_values, 5, axis=1),
            'percentile_95': np.percentile(self.portfolio_values, 95, axis=1),
            'initial_value': self.portfolio_history[0]['total_value'],
            'final_expected_value': np.mean(self.portfolio_values[-1, :]),
            'expected_return': (np.mean(self.portfolio_values[-1, :]) / self.portfolio_history[0]['total_value'] - 1) * 100,
            'volatility': np.std(self.portfolio_values[-1, :]) / np.mean(self.portfolio_values[-1, :]) * 100
        }
        
        # Prepare results
        results = {
            'portfolio_stats': portfolio_stats,
            'stock_simulations': self.simulation_results,
            'event_log': self.event_log,
            'action_log': self.action_log,
            'portfolio_history': self.portfolio_history
        }
        
        return results
    
    def plot_results(self, results: Dict[str, Any], show_stocks: bool = True, 
                    show_events: bool = True, show_actions: bool = True) -> None:
        """
        Plot simulation results.
        
        Args:
            results: Simulation results from run_simulation
            show_stocks: Whether to show individual stock lines
            show_events: Whether to highlight event periods
            show_actions: Whether to mark action points
        """
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio value
        portfolio_stats = results['portfolio_stats']
        days = np.arange(self.forecast_days)
        
        # Plot confidence interval
        plt.fill_between(days, 
                        portfolio_stats['percentile_05'], 
                        portfolio_stats['percentile_95'],
                        color='lightgray', alpha=0.5, label='90% Confidence Interval')
        
        # Plot expected value
        plt.plot(days, portfolio_stats['expected_value'], 'b-', linewidth=2, label='Expected Portfolio Value')
        
        # Plot individual stocks if requested
        if show_stocks:
            for ticker, sim_result in results['stock_simulations'].items():
                # Get initial quantity
                initial_qty = self.portfolio[ticker]['quantity']
                # Calculate weighted contribution to portfolio
                weighted_price = sim_result['expected_price'] * initial_qty
                plt.plot(days, weighted_price, '--', alpha=0.5, label=f"{ticker} (qty={initial_qty})")
        
        # Highlight event periods if requested
        if show_events:
            for event in results['event_log']:
                event_day = event['day']
                event_duration = event['duration']
                event_end = min(event_day + event_duration, self.forecast_days - 1)
                
                plt.axvspan(event_day, event_end, color='red', alpha=0.1)
                
                # Add event label
                y_pos = portfolio_stats['max_value'][event_day] * 0.95
                plt.text(event_day + (event_end - event_day) / 2, y_pos, 
                        event['type'], fontsize=8, ha='center', 
                        bbox=dict(facecolor='white', alpha=0.7))
        
        # Mark action points if requested
        if show_actions:
            for action in results['action_log']:
                action_day = action['day']
                action_type = action['type']
                
                if action_day < self.forecast_days:
                    y_pos = portfolio_stats['expected_value'][action_day]
                    plt.scatter(action_day, y_pos, marker='^', color='green', s=100, zorder=10)
                    plt.text(action_day, y_pos * 1.05, action_type, fontsize=8, ha='center', rotation=45)
        
        # Add labels and legend
        plt.title("Monte Carlo Portfolio Simulation (1 Year Forecast)")
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        # Show plot
        plt.show()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a detailed report of the simulation results.
        
        Args:
            results: Simulation results from run_simulation
            
        Returns:
            String containing the report
        """
        portfolio_stats = results['portfolio_stats']
        
        # Format currency values
        initial_value = f"₹{portfolio_stats['initial_value']:,.2f}"
        final_value = f"₹{portfolio_stats['final_expected_value']:,.2f}"
        min_value = f"₹{portfolio_stats['min_value'][-1]:,.2f}"
        max_value = f"₹{portfolio_stats['max_value'][-1]:,.2f}"
        
        # Build report
        report = []
        report.append("# Portfolio Monte Carlo Simulation Report\n")
        
        report.append("## Summary Statistics")
        report.append(f"- Initial Portfolio Value: {initial_value}")
        report.append(f"- Expected Final Value: {final_value}")
        report.append(f"- Expected Return: {portfolio_stats['expected_return']:.2f}%")
        report.append(f"- Portfolio Volatility: {portfolio_stats['volatility']:.2f}%")
        report.append(f"- Worst Case (5th percentile): {min_value}")
        report.append(f"- Best Case (95th percentile): {max_value}\n")
        
        report.append("## Events Log")
        for event in sorted(results['event_log'], key=lambda x: x['day']):
            day = event['day']
            month = day // 21 + 1  # Approximate month (21 trading days per month)
            report.append(f"- Month {month} (Day {day}): {event['type']} for {event['duration']} days - {event['description']}")
        report.append("")
        
        report.append("## Actions Log")
        for action in sorted(results['action_log'], key=lambda x: x['day']):
            day = action['day']
            month = day // 21 + 1  # Approximate month
            action_details = action['details']
            
            if action['type'] == 'add_lumpsum_all':
                report.append(f"- Month {month} (Day {day}): Added ₹{action_details['amount']:,.2f} lumpsum across all stocks")
            elif action['type'] == 'add_lumpsum_stock':
                report.append(f"- Month {month} (Day {day}): Added ₹{action_details['amount']:,.2f} to {action_details['ticker']}")
            elif action['type'] == 'book_profit':
                report.append(f"- Month {month} (Day {day}): Booked {action_details['percentage']}% profit from {action_details['ticker']}")
            elif action['type'] == 'close_position':
                report.append(f"- Month {month} (Day {day}): Closed position in {action_details['ticker']}")
            else:
                report.append(f"- Month {month} (Day {day}): {action['type']}")
        report.append("")
        
        report.append("## Stock Performance")
        for ticker, details in self.portfolio.items():
            sim_result = results['stock_simulations'][ticker]
            initial_price = details['current_price']
            final_price = sim_result['expected_price'][-1]
            return_pct = (final_price / initial_price - 1) * 100
            
            report.append(f"### {ticker} ({details.get('sector', 'Unknown Sector')})")
            report.append(f"- Initial Price: ₹{initial_price:.2f}")
            report.append(f"- Expected Final Price: ₹{final_price:.2f}")
            report.append(f"- Expected Return: {return_pct:.2f}%")
            report.append(f"- Quantity: {details['quantity']}")
            report.append(f"- Initial Value: ₹{(details['quantity'] * initial_price):,.2f}")
            report.append(f"- Expected Final Value: ₹{(details['quantity'] * final_price):,.2f}")
            report.append("")
        
        return "\n".join(report)


def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch stock information from Yahoo Finance including sector, current price, and analyst targets.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock information
    """
    try:
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price
        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
        if current_price is None:
            # Try to get the latest price from history
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                raise ValueError(f"Could not fetch current price for {ticker}")
        
        # Get sector
        sector = info.get('sector', 'Unknown')
        
        # Get analyst targets
        target_mean = info.get('targetMeanPrice', None)
        target_high = info.get('targetHighPrice', None)
        target_low = info.get('targetLowPrice', None)
        
        # If analyst targets are not available, estimate them
        if target_mean is None or target_high is None or target_low is None:
            # Estimate targets based on current price with some variance
            target_mean = current_price * 1.15  # 15% growth expectation
            target_high = current_price * 1.3   # 30% upside
            target_low = current_price * 0.9    # 10% downside
        
        return {
            'sector': sector,
            'current_price': current_price,
            'analyst_targets': {
                'min': target_low,
                'avg': target_mean,
                'max': target_high
            }
        }
    except Exception as e:
        print(f"Error fetching info for {ticker}: {str(e)}")
        # Return default values
        return {
            'sector': 'Unknown',
            'current_price': 100,  # Default price
            'analyst_targets': {
                'min': 90,
                'avg': 115,
                'max': 130
            }
        }

def simulate_portfolio_monte_carlo(
    portfolio_data: Dict[str, Dict[str, Any]],
    events: List[Dict[str, Any]] = None,
    actions: List[Dict[str, Any]] = None,
    forecast_days: int = 252,
    num_simulations: int = 1000,
    seed: int = None,
    plot_results: bool = False,
    fetch_info: bool = True
) -> Dict[str, Any]:
    """
    Run a Monte Carlo simulation on a stock portfolio with events and actions.
    
    Args:
        portfolio_data: Dictionary with stock details including:
                      - ticker: Stock symbol as key
                      - quantity: Number of shares
                      - buy_price: Purchase price per share
                      Optional (will be fetched if not provided and fetch_info=True):
                      - sector: Stock sector
                      - current_price: Current price per share
                      - analyst_targets: Dict with min, avg, max price targets
        events: List of event dictionaries, each containing:
                - type: Event type (e.g., "market_crash", "rbi_rate_cut")
                - day/month: Optional specific timing
                - duration: Optional specific duration
                - description: Optional description
        actions: List of action dictionaries, each containing:
                - type: Action type (e.g., "add_lumpsum_all", "book_profit")
                - Various parameters depending on action type
        forecast_days: Number of trading days to simulate (default: 252 = 1 year)
        num_simulations: Number of Monte Carlo simulations to run
        seed: Random seed for reproducibility (default: None for random seed)
        plot_results: Whether to display plots
        fetch_info: Whether to fetch missing stock information from Yahoo Finance
        
    Returns:
        Dictionary with simulation results
    """
    # Set default seed if not provided
    if seed is None:
        seed = int(datetime.datetime.now().timestamp())
    
    # Set default empty lists
    if events is None:
        events = []
    if actions is None:
        actions = []
    
    # Fetch missing stock information if requested
    if fetch_info:
        enriched_portfolio = {}
        for ticker, details in portfolio_data.items():
            enriched_details = details.copy()
            
            # Check if we need to fetch any information
            needs_info = ('sector' not in details or 
                         'current_price' not in details or 
                         'analyst_targets' not in details)
            
            if needs_info:
                print(f"Fetching information for {ticker}...")
                stock_info = fetch_stock_info(ticker)
                
                # Add missing information
                if 'sector' not in enriched_details:
                    enriched_details['sector'] = stock_info['sector']
                if 'current_price' not in enriched_details:
                    enriched_details['current_price'] = stock_info['current_price']
                if 'analyst_targets' not in enriched_details:
                    enriched_details['analyst_targets'] = stock_info['analyst_targets']
            
            enriched_portfolio[ticker] = enriched_details
        
        # Use the enriched portfolio
        portfolio_data = enriched_portfolio
    
    # Initialize simulator
    simulator = PortfolioMonteCarloSimulator(
        portfolio=portfolio_data,
        forecast_days=forecast_days,
        num_simulations=num_simulations,
        seed=seed
    )
    
    # Run simulation
    results = simulator.run_simulation(events, actions)
    
    # # Generate report
    # report = simulator.generate_report(results)
    # results['report'] = report
    
    # # Plot if requested
    # if plot_results:
    #     simulator.plot_results(results)
    
    return results


# Example usage
if __name__ == "__main__":
    # Sample portfolio data
    portfolio = {
        "RELIANCE.NS": {
            "quantity": 10,
            "buy_price": 2500
        },
        "TCS.NS": {
            "quantity": 5,
            "buy_price": 3200
        },
        "HDFCBANK.NS": {
            "quantity": 20,
            "buy_price": 1600
        },
        "INFY.NS": {
            "quantity": 15,
            "buy_price": 1400
        }
    }
    
    # Sample events
    events = [
        {"type": "market_correction", "month": 3},
        {"type": "rbi_rate_cut", "month": 6},
        {"type": "bull_run", "month": 9}
    ]
    
    # Sample actions
    actions = [
        {"type": "add_lumpsum_all", "amount": 50000, "month": 4},
        {"type": "add_lumpsum_stock", "ticker": "TCS.NS", "amount": 30000, "month": 7},
        {"type": "book_profit", "ticker": "RELIANCE.NS", "percentage": 30, "month": 10}
    ]
    
    # Run simulation
    results = simulate_portfolio_monte_carlo(
        portfolio_data=portfolio,
        events=events,
        actions=actions,
        forecast_days=252,  # 1 year
        num_simulations=500,
        plot_results=True
    )
    
    print(results)
