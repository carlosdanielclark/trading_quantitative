"""
Módulo de backtesting para estrategias de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """
    Representa una operación de trading
    """
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' or 'short'
    entry_reason: str
    exit_reason: Optional[str]
    commission: float
    
    @property
    def is_open(self) -> bool:
        """Verifica si la operación está abierta"""
        return self.exit_time is None
    
    @property
    def pnl(self) -> float:
        """Calcula el P&L de la operación"""
        if self.is_open:
            return 0.0
        
        if self.side == 'long':
            return (self.exit_price - self.entry_price) * self.quantity - self.commission
        else:  # short
            return (self.entry_price - self.exit_price) * self.quantity - self.commission
    
    @property
    def return_pct(self) -> float:
        """Calcula el retorno porcentual"""
        if self.is_open:
            return 0.0
        
        invested = self.entry_price * self.quantity
        return (self.pnl / invested) * 100

@dataclass
class BacktestResults:
    """
    Resultados del backtesting
    """
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    initial_capital: float
    final_capital: float
    
    @property
    def total_trades(self) -> int:
        """Número total de operaciones"""
        return len([t for t in self.trades if not t.is_open])
    
    @property
    def winning_trades(self) -> int:
        """Número de operaciones ganadoras"""
        return len([t for t in self.trades if not t.is_open and t.pnl > 0])
    
    @property
    def losing_trades(self) -> int:
        """Número de operaciones perdedoras"""
        return len([t for t in self.trades if not t.is_open and t.pnl < 0])
    
    @property
    def win_rate(self) -> float:
        """Tasa de acierto"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def total_return(self) -> float:
        """Retorno total en porcentaje"""
        return ((self.final_capital / self.initial_capital) - 1) * 100
    
    @property
    def max_drawdown(self) -> float:
        """Máximo drawdown"""
        if self.equity_curve.empty:
            return 0.0
        
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min() * 100
    
    @property
    def sharpe_ratio(self) -> float:
        """Ratio de Sharpe"""
        if self.returns.empty or self.returns.std() == 0:
            return 0.0
        
        return self.returns.mean() / self.returns.std() * np.sqrt(252)  # Anualizado
    
    @property
    def profit_factor(self) -> float:
        """Factor de beneficio"""
        winning_pnl = sum([t.pnl for t in self.trades if not t.is_open and t.pnl > 0])
        losing_pnl = abs(sum([t.pnl for t in self.trades if not t.is_open and t.pnl < 0]))
        
        if losing_pnl == 0:
            return float('inf') if winning_pnl > 0 else 0.0
        
        return winning_pnl / losing_pnl

class BacktestEngine:
    """
    Motor de backtesting para estrategias de trading
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """
        Inicializa el motor de backtesting
        
        Args:
            initial_capital: Capital inicial
            commission: Comisión por operación (como porcentaje)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.current_capital = initial_capital
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
        logger.info(f"BacktestEngine inicializado con capital: ${initial_capital}")

    def run_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                    position_size: float = 0.1) -> BacktestResults:
        """
        Ejecuta el backtesting
        
        Args:
            data: DataFrame con datos OHLCV
            signals: Serie con señales de trading (1=compra, -1=venta, 0=mantener)
            position_size: Tamaño de posición como porcentaje del capital
            
        Returns:
            Resultados del backtesting
        """
        logger.info("Iniciando backtesting...")
        
        # Validar datos
        if data.empty or signals.empty:
            raise ValueError("Datos o señales están vacíos")
        
        # Alinear datos y señales
        aligned_data = data.loc[signals.index]
        
        # Resetear estado
        self.current_capital = self.initial_capital
        self.trades = []
        self.open_trades = []
        self.equity_curve = []
        self.timestamps = []
        
        # Procesar cada punto temporal
        for timestamp, signal in signals.items():
            if timestamp not in aligned_data.index:
                continue
                
            current_price = aligned_data.loc[timestamp, 'close']
            
            # Procesar señal
            if signal == 1:  # Señal de compra
                self._process_buy_signal(timestamp, current_price, position_size)
            elif signal == -1:  # Señal de venta
                self._process_sell_signal(timestamp, current_price, "signal")
            
            # Actualizar curva de equity
            self._update_equity_curve(timestamp, current_price)
        
        # Cerrar operaciones abiertas
        if self.open_trades:
            final_price = aligned_data.iloc[-1]['close']
            final_timestamp = aligned_data.index[-1]
            
            for trade in self.open_trades.copy():
                self._close_trade(trade, final_timestamp, final_price, "end_of_data")
        
        # Crear resultados
        results = self._create_results()
        
        logger.info(f"Backtesting completado. Operaciones: {results.total_trades}")
        return results

    def _process_buy_signal(self, timestamp: datetime, price: float, position_size: float):
        """
        Procesa una señal de compra
        
        Args:
            timestamp: Momento de la señal
            price: Precio actual
            position_size: Tamaño de posición
        """
        # Cerrar posiciones cortas si existen
        for trade in self.open_trades.copy():
            if trade.side == 'short':
                self._close_trade(trade, timestamp, price, "signal_reversal")
        
        # Abrir nueva posición larga si hay capital
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
                    commission=commission_cost
                )
                
                self.open_trades.append(trade)
                self.current_capital -= investment
                
                logger.debug(f"Operación larga abierta: {quantity:.4f} @ ${price:.2f}")

    def _process_sell_signal(self, timestamp: datetime, price: float, reason: str):
        """
        Procesa una señal de venta
        
        Args:
            timestamp: Momento de la señal
            price: Precio actual
            reason: Razón de la venta
        """
        # Cerrar posiciones largas
        for trade in self.open_trades.copy():
            if trade.side == 'long':
                self._close_trade(trade, timestamp, price, reason)

    def _close_trade(self, trade: Trade, timestamp: datetime, price: float, reason: str):
        """
        Cierra una operación
        
        Args:
            trade: Operación a cerrar
            timestamp: Momento de cierre
            price: Precio de cierre
            reason: Razón del cierre
        """
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = reason
        
        # Calcular proceeds
        proceeds = trade.quantity * price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost
        
        # Actualizar capital
        self.current_capital += net_proceeds
        
        # Actualizar comisión total
        trade.commission += commission_cost
        
        # Mover a operaciones cerradas
        self.open_trades.remove(trade)
        self.trades.append(trade)
        
        logger.debug(f"Operación cerrada: {trade.side} P&L: ${trade.pnl:.2f}")

    def _update_equity_curve(self, timestamp: datetime, current_price: float):
        """
        Actualiza la curva de equity
        
        Args:
            timestamp: Momento actual
            current_price: Precio actual
        """
        # Calcular valor de posiciones abiertas
        open_positions_value = 0
        for trade in self.open_trades:
            if trade.side == 'long':
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:  # short
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
            
            open_positions_value += trade.entry_price * trade.quantity + unrealized_pnl
        
        # Equity total
        total_equity = self.current_capital + open_positions_value
        
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)

    def _create_results(self) -> BacktestResults:
        """
        Crea el objeto de resultados
        
        Returns:
            Resultados del backtesting
        """
        # Crear series
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        returns_series = equity_series.pct_change().dropna()
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_series,
            returns=returns_series,
            initial_capital=self.initial_capital,
            final_capital=self.equity_curve[-1] if self.equity_curve else self.initial_capital
        )

    def get_performance_metrics(self, results: BacktestResults) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento
        
        Args:
            results: Resultados del backtesting
            
        Returns:
            Diccionario con métricas
        """
        return {
            'total_trades': results.total_trades,
            'winning_trades': results.winning_trades,
            'losing_trades': results.losing_trades,
            'win_rate': results.win_rate,
            'total_return': results.total_return,
            'max_drawdown': results.max_drawdown,
            'sharpe_ratio': results.sharpe_ratio,
            'profit_factor': results.profit_factor,
            'initial_capital': results.initial_capital,
            'final_capital': results.final_capital,
        }
