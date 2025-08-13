import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime

class FinancialVisualizer:
    """
    Visualize quantum financial system components and metrics.
    
    Features:
    - Portfolio visualization
    - Risk analysis plots
    - Trading signals visualization
    - Performance metrics
    - Market data analysis
    - Interactive dashboards
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # Set style
        plt.style.use('seaborn')
        
    def plot_portfolio_composition(self,
                                 portfolio: Dict[str, Any],
                                 title: str = "Portfolio Composition",
                                 save: bool = False) -> None:
        """
        Plot portfolio asset allocation.
        
        Args:
            portfolio: Portfolio state
            title: Plot title
            save: Whether to save plot
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot asset allocation
        positions = portfolio['positions']
        values = [amount * portfolio['prices'][asset]
                 for asset, amount in positions.items()]
        labels = list(positions.keys())
        
        ax1.pie(values, labels=labels, autopct='%1.1f%%')
        ax1.set_title("Asset Allocation")
        
        # Plot historical returns
        returns = portfolio['returns']
        cumulative_returns = np.cumprod(1 + np.array(returns))
        
        ax2.plot(cumulative_returns)
        ax2.set_title("Cumulative Returns")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Return")
        
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_risk_metrics(self,
                         risk_metrics: Dict[str, Any],
                         title: str = "Risk Analysis",
                         save: bool = False) -> None:
        """
        Plot risk analysis metrics.
        
        Args:
            risk_metrics: Risk metrics
            title: Plot title
            save: Whether to save plot
        """
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot VaR
        var_data = risk_metrics['var']
        ax1.bar(['95% VaR', '99% VaR'],
                [var_data['var_95'], var_data['var_99']])
        ax1.set_title("Value at Risk")
        
        # Plot stress test results
        stress_results = risk_metrics['stress_results']
        scenarios = list(stress_results.keys())
        impacts = [result['value_impact']
                  for result in stress_results.values()]
        
        ax2.bar(scenarios, impacts)
        ax2.set_title("Stress Test Results")
        plt.xticks(rotation=45)
        
        # Plot risk measures
        measures = risk_metrics['risk_measures']
        ax3.bar(measures.keys(), measures.values())
        ax3.set_title("Risk Measures")
        plt.xticks(rotation=45)
        
        # Plot correlation matrix
        correlation_matrix = risk_metrics['correlation_matrix']
        sns.heatmap(correlation_matrix, ax=ax4)
        ax4.set_title("Asset Correlations")
        
        plt.tight_layout()
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_trading_signals(self,
                           signals: Dict[str, Any],
                           title: str = "Trading Signals",
                           save: bool = False) -> None:
        """
        Plot trading signals and execution.
        
        Args:
            signals: Trading signals
            title: Plot title
            save: Whether to save plot
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot signals
        assets = list(signals.keys())
        actions = [signal['action'] for signal in signals.values()]
        amounts = [signal['amount'] for signal in signals.values()]
        
        colors = ['g' if action == 'buy' else 'r'
                 for action in actions]
        
        ax1.bar(assets, amounts, color=colors)
        ax1.set_title("Trading Signals")
        ax1.set_ylabel("Amount")
        
        # Plot execution costs
        costs = [signal['expected_cost'] for signal in signals.values()]
        ax2.bar(assets, costs)
        ax2.set_title("Expected Execution Costs")
        ax2.set_ylabel("Cost")
        
        plt.tight_layout()
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_performance_metrics(self,
                               metrics: Dict[str, List[float]],
                               title: str = "Performance Metrics",
                               save: bool = False) -> None:
        """
        Plot performance metrics history.
        
        Args:
            metrics: Performance metrics
            title: Plot title
            save: Whether to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values)
            ax.set_title(f"{metric_name} History")
            ax.set_xlabel("Time")
            ax.set_ylabel(metric_name)
            
        plt.tight_layout()
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_market_analysis(self,
                           market_data: Dict[str, pd.DataFrame],
                           title: str = "Market Analysis",
                           save: bool = False) -> None:
        """
        Plot market data analysis.
        
        Args:
            market_data: Market data by asset
            title: Plot title
            save: Whether to save plot
        """
        n_assets = len(market_data)
        fig, axes = plt.subplots(n_assets, 3, figsize=(15, 5*n_assets))
        
        for i, (asset, data) in enumerate(market_data.items()):
            # Price plot
            axes[i, 0].plot(data.index, data['price'])
            axes[i, 0].set_title(f"{asset} Price")
            
            # Volume plot
            axes[i, 1].bar(data.index, data['volume'])
            axes[i, 1].set_title(f"{asset} Volume")
            
            # Volatility plot
            volatility = data['price'].pct_change().rolling(20).std()
            axes[i, 2].plot(data.index, volatility)
            axes[i, 2].set_title(f"{asset} Volatility")
            
        plt.tight_layout()
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def create_interactive_dashboard(self,
                                   portfolio: Dict[str, Any],
                                   market_data: Dict[str, pd.DataFrame],
                                   title: str = "Portfolio Dashboard") -> None:
        """
        Create interactive portfolio dashboard.
        
        Args:
            portfolio: Portfolio state
            market_data: Market data
            title: Dashboard title
        """
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=2,
                           subplot_titles=("Portfolio Value",
                                         "Asset Allocation",
                                         "Returns Distribution",
                                         "Risk Metrics",
                                         "Asset Prices",
                                         "Trading Activity"))
                                         
        # Portfolio value
        portfolio_values = np.cumprod(1 + np.array(portfolio['returns']))
        fig.add_trace(
            go.Scatter(y=portfolio_values, name="Portfolio Value"),
            row=1, col=1
        )
        
        # Asset allocation
        positions = portfolio['positions']
        values = [amount * portfolio['prices'][asset]
                 for asset, amount in positions.items()]
        fig.add_trace(
            go.Pie(labels=list(positions.keys()),
                  values=values,
                  name="Allocation"),
            row=1, col=2
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=portfolio['returns'],
                        name="Returns Distribution"),
            row=2, col=1
        )
        
        # Risk metrics
        risk_metrics = ['Sharpe Ratio', 'Max Drawdown', 'Volatility']
        metric_values = [portfolio['sharpe_ratio'],
                        portfolio['max_drawdown'],
                        np.std(portfolio['returns'])]
        fig.add_trace(
            go.Bar(x=risk_metrics,
                  y=metric_values,
                  name="Risk Metrics"),
            row=2, col=2
        )
        
        # Asset prices
        for asset, data in market_data.items():
            fig.add_trace(
                go.Scatter(x=data.index,
                          y=data['price'],
                          name=f"{asset} Price"),
                row=3, col=1
            )
            
        # Trading activity
        trades = portfolio.get('trades', [])
        if trades:
            trade_times = [t['timestamp'] for t in trades]
            trade_amounts = [t['amount'] for t in trades]
            fig.add_trace(
                go.Scatter(x=trade_times,
                          y=trade_amounts,
                          mode='markers',
                          name="Trades"),
                row=3, col=2
            )
            
        # Update layout
        fig.update_layout(height=1000, title_text=title)
        fig.show()
        
    def animate_portfolio_evolution(self,
                                  portfolio_history: List[Dict[str, Any]],
                                  interval: int = 100,
                                  title: str = "Portfolio Evolution",
                                  save: bool = False) -> None:
        """
        Create animation of portfolio evolution.
        
        Args:
            portfolio_history: List of portfolio states
            interval: Animation interval in milliseconds
            title: Animation title
            save: Whether to save plot
        """
        import matplotlib.animation as animation
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            portfolio = portfolio_history[frame]
            
            # Plot asset allocation
            positions = portfolio['positions']
            values = [amount * portfolio['prices'][asset]
                     for asset, amount in positions.items()]
            ax1.pie(values, labels=list(positions.keys()),
                   autopct='%1.1f%%')
            ax1.set_title("Asset Allocation")
            
            # Plot portfolio value
            values = [p['total_value']
                     for p in portfolio_history[:frame+1]]
            ax2.plot(values)
            ax2.set_title("Portfolio Value")
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(portfolio_history),
            interval=interval
        )
        
        plt.suptitle(title)
        
        if save and self.save_dir:
            anim.save(self.save_dir / f"{title.lower().replace(' ', '_')}.gif")
        else:
            plt.show()
