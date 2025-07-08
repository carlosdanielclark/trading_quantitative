import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.stats import norm
from fpdf import FPDF

logger = logging.getLogger("BacktestEngine")

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' or 'short'
    entry_reason: str
    exit_reason: Optional[str]
    commission: float
    slippage: float
    spread: float

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def pnl(self) -> float:
        if self.is_open:
            return 0.0
        if self.side == 'long':
            return (self.exit_price - self.entry_price - self.spread - self.slippage) * self.quantity - self.commission
        else:
            return (self.entry_price - self.exit_price - self.spread - self.slippage) * self.quantity - self.commission

    @property
    def return_pct(self) -> float:
        if self.is_open:
            return 0.0
        invested = self.entry_price * self.quantity
        return (self.pnl / invested) * 100 if invested > 0 else 0.0

@dataclass
class BacktestResults:
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    initial_capital: float
    final_capital: float
    drawdowns: pd.Series
    walk_forward_metrics: Dict[str, Any]

    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open])

    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open and t.pnl > 0])

    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open and t.pnl < 0])

    @property
    def win_rate(self) -> float:
        return (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0.0

    @property
    def total_return(self) -> float:
        return ((self.final_capital / self.initial_capital) - 1) * 100

    @property
    def max_drawdown(self) -> float:
        return self.drawdowns.min() * 100 if not self.drawdowns.empty else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if self.returns.empty or self.returns.std() == 0:
            return 0.0
        return (self.returns.mean() / self.returns.std()) * np.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        downside = self.returns[self.returns < 0]
        if len(downside) == 0:
            # No hay retornos negativos, Sortino es infinito o muy alto
            return float('inf')
        downside_std = downside.std()
        if downside_std == 0:
            # Desviación cero, evitar división por cero
            return float('inf')
        return (self.returns.mean() / downside_std) * np.sqrt(252)

    @property
    def profit_factor(self) -> float:
        winning_pnl = sum([t.pnl for t in self.trades if not t.is_open and t.pnl > 0])
        losing_pnl = abs(sum([t.pnl for t in self.trades if not t.is_open and t.pnl < 0]))
        if losing_pnl == 0:
            return float('inf') if winning_pnl > 0 else 0.0
        return winning_pnl / losing_pnl

    @property
    def cagr(self) -> float:
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        if years == 0:
            return 0.0
        return (self.final_capital / self.initial_capital) ** (1 / years) - 1

class BacktestEngine:
    def __init__(self, config: Dict[str, Any]):
        backtest_cfg = config.get('backtest', {})
        self.initial_capital = backtest_cfg.get('initial_capital', 10000)
        self.commission = backtest_cfg.get('commission', 0.001)
        self.spread = backtest_cfg.get('spread', 0.0)
        self.slippage = backtest_cfg.get('slippage', 0.0)
        self.current_capital = self.initial_capital
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def run_backtest(self, data: pd.DataFrame, signals: pd.Series,
                     position_size: float = 0.1,
                     walk_forward: bool = False,
                     walk_window: int = 252) -> BacktestResults:
        logger.info("Iniciando backtesting...")
        if data.empty or signals.empty:
            raise ValueError("Datos o señales están vacíos")
        aligned_data = data.loc[signals.index]
        self.current_capital = self.initial_capital
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.timestamps = []
        drawdowns = []
        walk_forward_metrics = {}

        for timestamp, signal in signals.items():
            if timestamp not in aligned_data.index:
                continue
            current_price = aligned_data.loc[timestamp, 'close']
            # Simula spread y slippage
            fill_price = current_price + self.spread if signal == 1 else current_price - self.spread if signal == -1 else current_price
            fill_price += np.random.normal(0, self.slippage) if self.slippage > 0 else 0

            if signal == 1:
                self._process_buy_signal(timestamp, fill_price, position_size)
            elif signal == -1:
                self._process_sell_signal(timestamp, fill_price, "signal")
            self._update_equity_curve(timestamp, fill_price)
            drawdowns.append(self._compute_drawdown())

        # Cierra operaciones abiertas al final
        if self.open_trades:
            final_price = aligned_data.iloc[-1]['close']
            final_timestamp = aligned_data.index[-1]
            for trade in self.open_trades.copy():
                self._close_trade(trade, final_timestamp, final_price, "end_of_data")

        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        returns_series = equity_series.pct_change().dropna()
        drawdown_series = pd.Series(drawdowns, index=self.timestamps)

        # Walk-forward analysis
        if walk_forward:
            walk_forward_metrics = self._walk_forward_analysis(equity_series, walk_window)

        results = BacktestResults(
            trades=self.trades,
            equity_curve=equity_series,
            returns=returns_series,
            initial_capital=self.initial_capital,
            final_capital=self.equity_curve[-1] if self.equity_curve else self.initial_capital,
            drawdowns=drawdown_series,
            walk_forward_metrics=walk_forward_metrics
        )
        logger.info(f"Backtesting completado. Operaciones: {results.total_trades}")
        self._detect_biases(results, data)
        return results

    def _process_buy_signal(self, timestamp, price, position_size):
        for trade in self.open_trades.copy():
            if trade.side == 'short':
                self._close_trade(trade, timestamp, price, "signal_reversal")
        if self.current_capital > 0:
            investment = self.current_capital * position_size
            commission_cost = investment * self.commission
            quantity = (investment - commission_cost) / price
            if quantity > 0:
                trade = Trade(
                    entry_time=timestamp,
                    exit_time=None,
                    entry_price=price,
                    exit_price=None,
                    quantity=quantity,
                    side='long',
                    entry_reason='signal',
                    exit_reason=None,
                    commission=commission_cost,
                    slippage=self.slippage,
                    spread=self.spread
                )
                self.open_trades.append(trade)
                self.current_capital -= investment

    def _process_sell_signal(self, timestamp, price, reason):
        for trade in self.open_trades.copy():
            if trade.side == 'long':
                self._close_trade(trade, timestamp, price, reason)

    def _close_trade(self, trade, timestamp, price, reason):
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = reason
        proceeds = trade.quantity * price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost
        self.current_capital += net_proceeds
        trade.commission += commission_cost
        self.open_trades.remove(trade)
        self.trades.append(trade)

    def _update_equity_curve(self, timestamp, current_price):
        open_positions_value = sum(
            (current_price - t.entry_price) * t.quantity if t.side == 'long'
            else (t.entry_price - current_price) * t.quantity
            for t in self.open_trades
        )
        total_equity = self.current_capital + open_positions_value
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)

    def _compute_drawdown(self):
        if not self.equity_curve:
            return 0.0
        peak = max(self.equity_curve)
        dd = (self.equity_curve[-1] - peak) / peak if peak > 0 else 0.0
        return dd

    def _walk_forward_analysis(self, equity_curve, window):
        metrics = []
        for start in range(0, len(equity_curve) - window, window):
            segment = equity_curve.iloc[start:start+window]
            ret = (segment.iloc[-1] / segment.iloc[0] - 1) if segment.iloc[0] > 0 else 0.0
            dd = (segment.min() - segment.max()) / segment.max() if segment.max() > 0 else 0.0
            metrics.append({'return': ret, 'drawdown': dd})
        return metrics

    def _detect_biases(self, results: BacktestResults, data: pd.DataFrame):
        # Look-ahead bias: compara timestamps de señales con datos futuros
        if any(t.entry_time >= t.exit_time for t in results.trades if not t.is_open):
            logger.warning("¡Posible look-ahead bias detectado! Revisa la alineación de señales y datos.")
        # Data snooping: advierte si se usaron demasiados parámetros o señales optimizadas
        if len(results.trades) > 0 and results.win_rate > 90:
            logger.warning("¡Alerta de data snooping! Win rate inusualmente alto.")

    def get_performance_metrics(self, results: BacktestResults, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
        metrics = {
            'total_trades': results.total_trades,
            'winning_trades': results.winning_trades,
            'losing_trades': results.losing_trades,
            'win_rate': results.win_rate,
            'total_return': results.total_return,
            'cagr': results.cagr * 100,
            'max_drawdown': results.max_drawdown,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'profit_factor': results.profit_factor,
            'initial_capital': results.initial_capital,
            'final_capital': results.final_capital,
        }
        # Intervalos de confianza (bootstrap)
        returns = results.returns
        if not returns.empty:
            mean = returns.mean()
            std = returns.std()
            ci = norm.interval(0.95, loc=mean, scale=std/np.sqrt(len(returns)))
            metrics['mean_return_ci_95'] = ci
        # Correlación con benchmark
        if benchmark is not None:
            aligned = results.returns.align(benchmark, join='inner')[0]
            metrics['correlation_benchmark'] = aligned.corr(benchmark)
        return metrics

    def monte_carlo_analysis(self, results: BacktestResults, n_sim: int = 1000) -> Dict[str, Any]:
        returns = results.returns.dropna().values
        simulations = []
        for _ in range(n_sim):
            sim = np.random.choice(returns, size=len(returns), replace=True)
            sim_curve = np.cumprod(1 + sim)
            simulations.append(sim_curve[-1])
        ci = np.percentile(simulations, [2.5, 97.5])
        return {'final_equity_ci_95': ci}

    def plot_equity_curve(self, results: BacktestResults):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.equity_curve.index, y=results.equity_curve, name='Equity Curve'))
        fig.add_trace(go.Scatter(x=results.drawdowns.index, y=results.drawdowns * results.equity_curve, name='Drawdown', fill='tozeroy', opacity=0.3))
        fig.update_layout(title='Equity Curve with Drawdown', xaxis_title='Time', yaxis_title='Equity')
        fig.show()

    def generate_pdf_report(self, results: BacktestResults, metrics: Dict[str, Any], filename: str = "backtest_report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Backtest Report", ln=True, align='C')
        for k, v in metrics.items():
            pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
        pdf.output(filename)
        logger.info(f"Reporte PDF generado: {filename}")