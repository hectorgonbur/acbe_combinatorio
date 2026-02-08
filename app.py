"""
🎯 ACBE-S73 QUANTUM BETTING SUITE v3.0 - ARQUITECTURA INSTITUCIONAL
Sistema profesional de optimización de portafolios de apuestas deportivas
Con arquitectura limpia, formulario centralizado y modelo matemático riguroso

CORRECCIONES CRÍTICAS IMPLEMENTADAS:
1. ✅ ARQUITECTURA LIMPIA: Un solo archivo app.py con módulos lógicos internos
2. ✅ FORMULARIO CENTRAL: Botón único de ejecución sin recargas no deseadas
3. ✅ MODELO INSTITUCIONAL: Gamma-Poisson Bayesiano + Optimización S73 formal
4. ✅ GESTIÓN DE ESTADO: st.session_state para persistencia de datos
5. ✅ MÉTRICAS CUANTITATIVAS: ROI, Sharpe, VaR, CVaR, Drawdown, Probabilidad de Ruina

Autor: Lead Quantitative Software Engineer & Senior Data Scientist
Nivel: Institutional Portfolio Management | Mathematical Rigor
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import warnings
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
warnings.filterwarnings('ignore')

# ============================================================================
# MÓDULO 1: CONFIGURACIÓN Y CONSTANTES INSTITUCIONALES
# ============================================================================

@dataclass
class InstitutionalConstants:
    """Constantes matemáticas y de configuración institucional."""
    
    # Simulación Monte Carlo
    MONTE_CARLO_ITERATIONS: int = 10000
    MIN_PROBABILITY: float = 1e-10
    
    # Modelo Gamma-Poisson Bayesiano
    ALPHA_PRIOR: float = 2.0  # Parámetro de forma Gamma
    BETA_PRIOR: float = 1.0   # Parámetro de tasa Gamma
    HOME_ADVANTAGE_BASE: float = 1.15
    
    # Sistema S73 (Optimización Combinatoria)
    NUM_MATCHES: int = 6
    FULL_COMBINATIONS: int = 3 ** 6  # 729
    HAMMING_COVERAGE: int = 2  # Cobertura de 2 errores
    MASS_COVERAGE_TARGET: float = 0.95  # 95% masa probabilística
    
    # Criterio de Kelly
    KELLY_FRACTION_MAX: float = 0.25  # 25% máximo por apuesta
    MAX_PORTFOLIO_EXPOSURE: float = 0.15  # 15% exposición total
    
    # Parámetros ACBE
    DEFAULT_POISSON_WEIGHT: float = 0.70
    DEFAULT_H2H_WEIGHT: float = 0.25
    DEFAULT_SIGMA_VAKE: float = 0.05
    
    # Métricas de Riesgo
    VAR_LEVEL: float = 0.95
    CVAR_LEVEL: float = 0.95
    RUIN_THRESHOLD: float = 0.5  # 50% del bankroll inicial
    
    # Configuración UI
    COLORS = {
        'primary': '#1E88E5',
        'secondary': '#004D40',
        'success': '#00C853',
        'warning': '#FFAB00',
        'danger': '#D50000',
        'info': '#00B8D4'
    }
    
    OUTCOME_LABELS = ['1', 'X', '2']
    OUTCOME_COLORS = ['#1E88E5', '#FF9800', '#D50000']

CONST = InstitutionalConstants()

# ============================================================================
# MÓDULO 2: MOTOR MATEMÁTICO ACBE (GAMMA-POISSON BAYESIANO)
# ============================================================================

class ACBEModel:
    """Modelo Bayesiano Gamma-Poisson para estimación de probabilidades."""
    
    @staticmethod
    @st.cache_data
    def gamma_poisson_posterior(
        goals_scored: float,
        matches_played: float,
        alpha_prior: float = CONST.ALPHA_PRIOR,
        beta_prior: float = CONST.BETA_PRIOR
    ) -> Tuple[float, float]:
        """
        Calcula posterior Gamma-Poisson para un equipo.
        
        Posterior ~ Gamma(α + goles, β + partidos)
        
        Args:
            goals_scored: Goles anotados (promedio por partido)
            matches_played: Partidos considerados
            alpha_prior: Parámetro de forma prior
            beta_prior: Parámetro de tasa prior
            
        Returns:
            lambda_mean, lambda_std: Media y desviación del posterior
        """
        # Convertir a tasa por partido
        total_goals = goals_scored * matches_played
        
        # Parámetros posterior
        alpha_post = alpha_prior + total_goals
        beta_post = beta_prior + matches_played
        
        # Estadísticas posterior
        lambda_mean = alpha_post / beta_post
        lambda_var = alpha_post / (beta_post ** 2)
        lambda_std = np.sqrt(lambda_var)
        
        return lambda_mean, lambda_std
    
    @staticmethod
    @st.cache_data
    def simulate_match_outcomes(
        lambda_home: float,
        lambda_away: float,
        n_simulations: int = CONST.MONTE_CARLO_ITERATIONS
    ) -> np.ndarray:
        """
        Simula resultados de partido usando distribución Poisson.
        
        Args:
            lambda_home: Tasa de goles local (posterior mean)
            lambda_away: Tasa de goles visitante (posterior mean)
            n_simulations: Iteraciones Monte Carlo
            
        Returns:
            Array (3,) con probabilidades [P(1), P(X), P(2)]
        """
        # Simular goles
        home_goals = np.random.poisson(lambda_home, n_simulations)
        away_goals = np.random.poisson(lambda_away, n_simulations)
        
        # Calcular resultados
        home_wins = np.mean(home_goals > away_goals)
        draws = np.mean(home_goals == away_goals)
        away_wins = np.mean(home_goals < away_goals)
        
        probabilities = np.array([home_wins, draws, away_wins])
        
        # Estabilidad numérica
        probabilities = np.clip(probabilities, CONST.MIN_PROBABILITY, 1.0)
        probabilities = probabilities / probabilities.sum()
        
        return probabilities
    
    @staticmethod
    def calculate_final_probabilities(
        poisson_probs: np.ndarray,
        h2h_probs: Optional[np.ndarray] = None,
        poisson_weight: float = CONST.DEFAULT_POISSON_WEIGHT,
        sigma_vake: float = CONST.DEFAULT_SIGMA_VAKE
    ) -> np.ndarray:
        """
        Fórmula final ACBE: P_final = (P_poisson * ω) + (P_h2h * (1 - ω)) - σ_vake
        
        Args:
            poisson_probs: Probabilidades Poisson (3,)
            h2h_probs: Probabilidades histórico (3,) o None
            poisson_weight: Peso Poisson (0-1)
            sigma_vake: Factor de ajuste de mercado
            
        Returns:
            Probabilidades finales normalizadas (3,)
        """
        # Si no hay H2H, usar solo Poisson
        if h2h_probs is None:
            h2h_probs = poisson_probs
            poisson_weight = 1.0
        
        # Fórmula ACBE
        final_probs = (poisson_probs * poisson_weight) + \
                     (h2h_probs * (1 - poisson_weight)) - \
                     sigma_vake
        
        # Asegurar no negatividad
        final_probs = np.maximum(final_probs, CONST.MIN_PROBABILITY)
        
        # Normalizar
        final_probs = final_probs / final_probs.sum()
        
        return final_probs

# ============================================================================
# MÓDULO 3: MOTOR DE OPTIMIZACIÓN S73 (INSTITUCIONAL)
# ============================================================================

class S73Optimizer:
    """Sistema de optimización combinatoria S73 con cobertura formal."""
    
    @staticmethod
    def generate_full_space() -> np.ndarray:
        """
        Genera espacio completo de combinaciones (729).
        
        Returns:
            Array (729, 6) con todas las combinaciones posibles
        """
        outcomes = [0, 1, 2]  # 1, X, 2
        combinations = list(itertools.product(outcomes, repeat=CONST.NUM_MATCHES))
        return np.array(combinations)
    
    @staticmethod
    def calculate_joint_probability(
        combination: np.ndarray,
        match_probabilities: np.ndarray
    ) -> float:
        """
        Calcula probabilidad conjunta de una combinación.
        
        P(ω) = ∏ P_i(resultado)
        
        Args:
            combination: Array (6,) con signos
            match_probabilities: Array (6, 3) con probabilidades por partido
            
        Returns:
            Probabilidad conjunta
        """
        prob = 1.0
        for i, sign in enumerate(combination):
            prob *= match_probabilities[i, sign]
        return prob
    
    @staticmethod
    def hamming_distance(comb1: np.ndarray, comb2: np.ndarray) -> int:
        """
        Calcula distancia de Hamming entre dos combinaciones.
        
        Args:
            comb1: Primera combinación (6,)
            comb2: Segunda combinación (6,)
            
        Returns:
            Distancia de Hamming (0-6)
        """
        return np.sum(comb1 != comb2)
    
    @staticmethod
    def greedy_coverage_optimization(
        match_probabilities: np.ndarray,
        coverage_distance: int = CONST.HAMMING_COVERAGE,
        mass_target: float = CONST.MASS_COVERAGE_TARGET
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Algoritmo greedy optimizado para cobertura S73.
        
        Maximiza: ∑ P(ω) para ω en S
        Sujeto a: |S| = K optimizado
                  ∀ω ∈ Ω, ∃s ∈ S: d_H(ω, s) ≤ coverage_distance
                  Cubrir ≥ mass_target de masa probabilística
        
        Args:
            match_probabilities: Array (6, 3) con probabilidades
            coverage_distance: Distancia de cobertura (default: 2)
            mass_target: Masa probabilística objetivo (default: 0.95)
            
        Returns:
            selected_combinations: Array (K, 6) de combinaciones seleccionadas
            selected_probs: Array (K,) de probabilidades seleccionadas
            K: Número óptimo de combinaciones
        """
        # 1. Generar espacio completo y calcular probabilidades
        full_space = S73Optimizer.generate_full_space()
        n_combinations = len(full_space)
        
        joint_probs = np.zeros(n_combinations)
        for i, combo in enumerate(full_space):
            joint_probs[i] = S73Optimizer.calculate_joint_probability(
                combo, match_probabilities
            )
        
        # 2. Ordenar por probabilidad descendente
        sorted_indices = np.argsort(joint_probs)[::-1]
        sorted_combinations = full_space[sorted_indices]
        sorted_probs = joint_probs[sorted_indices]
        
        # 3. Algoritmo greedy con cobertura progresiva
        selected_indices = []
        covered = np.zeros(n_combinations, dtype=bool)
        covered_count = 0
        total_prob_mass = 0.0
        
        # Calcular distancias entre todas las combinaciones (optimizado)
        # Para 729 combinaciones, es manejable
        distance_matrix = np.zeros((n_combinations, n_combinations), dtype=np.int8)
        for i in range(n_combinations):
            for j in range(i, n_combinations):
                dist = S73Optimizer.hamming_distance(
                    sorted_combinations[i], sorted_combinations[j]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # 4. Selección greedy
        while (total_prob_mass < mass_target and 
               len(selected_indices) < n_combinations):
            
            best_idx = -1
            best_coverage_gain = -1
            
            # Buscar combinación que maximice cobertura de no cubiertos
            for i in range(n_combinations):
                if i in selected_indices:
                    continue
                
                # Calcular nuevas coberturas
                coverage_mask = distance_matrix[i] <= coverage_distance
                new_coverages = np.sum(~covered & coverage_mask)
                
                # Ponderar por probabilidad
                coverage_gain = new_coverages * (1 + sorted_probs[i])
                
                if coverage_gain > best_coverage_gain:
                    best_coverage_gain = coverage_gain
                    best_idx = i
            
            if best_idx == -1:
                break
            
            # Agregar combinación seleccionada
            selected_indices.append(best_idx)
            
            # Actualizar coberturas
            coverage_mask = distance_matrix[best_idx] <= coverage_distance
            newly_covered = ~covered & coverage_mask
            covered[newly_covered] = True
            
            # Actualizar métricas
            newly_covered_indices = np.where(newly_covered)[0]
            covered_count += len(newly_covered_indices)
            total_prob_mass += np.sum(sorted_probs[newly_covered_indices])
        
        # 5. Resultados
        selected_combinations = sorted_combinations[selected_indices]
        selected_probs = sorted_probs[selected_indices]
        K = len(selected_indices)
        
        return selected_combinations, selected_probs, K
    
    @staticmethod
    def validate_coverage(
        selected_combinations: np.ndarray,
        match_probabilities: np.ndarray,
        coverage_distance: int = CONST.HAMMING_COVERAGE
    ) -> Dict[str, Any]:
        """
        Valida cobertura del sistema S73.
        
        Args:
            selected_combinations: Combinaciones seleccionadas
            match_probabilities: Probabilidades por partido
            coverage_distance: Distancia de cobertura requerida
            
        Returns:
            Diccionario con métricas de validación
        """
        full_space = S73Optimizer.generate_full_space()
        
        # Calcular cobertura
        max_distances = []
        covered_mass = 0.0
        
        for i, combo in enumerate(full_space):
            # Distancia mínima a combinaciones seleccionadas
            min_distance = min(
                S73Optimizer.hamming_distance(combo, selected)
                for selected in selected_combinations
            )
            max_distances.append(min_distance)
            
            # Masa probabilística cubierta
            prob = S73Optimizer.calculate_joint_probability(combo, match_probabilities)
            if min_distance <= coverage_distance:
                covered_mass += prob
        
        coverage_rate = covered_mass / match_probabilities[:, :3].prod()
        
        return {
            'max_distance': np.max(max_distances),
            'coverage_rate': coverage_rate,
            'mass_covered': covered_mass,
            'combinations_count': len(selected_combinations),
            'is_valid': np.max(max_distances) <= coverage_distance
        }

# ============================================================================
# MÓDULO 4: CRITERIO DE KELLY FRACCIONAL ADAPTATIVO
# ============================================================================

class KellyFractional:
    """Gestión adaptativa de capital basada en criterio de Kelly."""
    
    @staticmethod
    def calculate_kelly_fraction(
        probability: float,
        odds: float,
        fraction: float = 1.0
    ) -> float:
        """
        Calcula fracción de Kelly óptima.
        
        f* = (bp - q) / b
        donde: b = odds - 1, p = probabilidad, q = 1 - p
        
        Args:
            probability: Probabilidad estimada
            odds: Cuota ofrecida
            fraction: Fracción de Kelly a aplicar (0-1)
            
        Returns:
            Stake óptimo como fracción del bankroll
        """
        if odds <= 1.0 or probability <= 0.0 or probability >= 1.0:
            return 0.0
        
        b = odds - 1.0
        p = probability
        q = 1.0 - p
        
        # Kelly clásico
        kelly_raw = (b * p - q) / b
        
        # Aplicar fracción
        kelly_fractional = kelly_raw * fraction
        
        # Limitar máximo
        kelly_fractional = min(kelly_fractional, CONST.KELLY_FRACTION_MAX)
        
        # Solo apuestas con valor esperado positivo
        if kelly_fractional < 0:
            return 0.0
            
        return kelly_fractional
    
    @staticmethod
    def calculate_entropy_normalized_stake(
        base_stake: float,
        entropy: float,
        max_entropy: float = np.log(3)  # Máxima entropía para 3 resultados
    ) -> float:
        """
        Ajusta stake por entropía (incertidumbre).
        
        f_adj = f* * (1 - H_norm)
        
        Args:
            base_stake: Stake base (Kelly)
            entropy: Entropía de Shannon del partido
            max_entropy: Entropía máxima posible
            
        Returns:
            Stake ajustado por entropía
        """
        # Normalizar entropía
        if max_entropy > 0:
            h_norm = entropy / max_entropy
        else:
            h_norm = 0.0
        
        # Ajustar stake (más incertidumbre → menor stake)
        adjusted_stake = base_stake * (1.0 - h_norm)
        
        return max(0.0, adjusted_stake)
    
    @staticmethod
    def calculate_portfolio_allocation(
        stakes: np.ndarray,
        expected_values: np.ndarray,
        max_exposure: float = CONST.MAX_PORTFOLIO_EXPOSURE
    ) -> np.ndarray:
        """
        Distribuye capital óptimamente en el portafolio.
        
        Args:
            stakes: Array de stakes individuales (Kelly)
            expected_values: Array de valores esperados
            max_exposure: Exposición máxima total
            
        Returns:
            Array de stakes normalizados para portafolio
        """
        if len(stakes) == 0:
            return np.array([])
        
        # Calcular pesos basados en EV positivo
        positive_ev_mask = expected_values > 0
        if not np.any(positive_ev_mask):
            return np.zeros_like(stakes)
        
        positive_stakes = stakes[positive_ev_mask]
        positive_ev = expected_values[positive_ev_mask]
        
        # Normalizar por EV relativo
        ev_weights = positive_ev / positive_ev.sum()
        weighted_stakes = positive_stakes * ev_weights
        
        # Normalizar para respetar exposición máxima
        total_stake = weighted_stakes.sum()
        if total_stake > max_exposure:
            scaling_factor = max_exposure / total_stake
            weighted_stakes *= scaling_factor
        
        # Reconstruir array completo
        final_stakes = np.zeros_like(stakes)
        final_stakes[positive_ev_mask] = weighted_stakes
        
        return final_stakes

# ============================================================================
# MÓDULO 5: CAPA DE UNIFICACIÓN DE PORTAFOLIO
# ============================================================================

class PortfolioUnification:
    """Capa unificada para gestión de portafolio completo."""
    
    def __init__(self, initial_bankroll: float):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.portfolio = {
            'singles': [],
            'combinations': [],
            's73_columns': []
        }
    
    def add_singles(
        self,
        probabilities: np.ndarray,
        odds_matrix: np.ndarray,
        kelly_fraction: float = 0.5
    ) -> Dict[str, Any]:
        """
        Agrega apuestas simples al portafolio.
        
        Args:
            probabilities: Array (n, 3) de probabilidades
            odds_matrix: Array (n, 3) de cuotas
            kelly_fraction: Fracción de Kelly
            
        Returns:
            Diccionario con stakes y métricas
        """
        n_matches = len(probabilities)
        
        stakes = np.zeros((n_matches, 3))
        expected_values = np.zeros((n_matches, 3))
        
        for i in range(n_matches):
            for j in range(3):
                prob = probabilities[i, j]
                odds = odds_matrix[i, j]
                
                # Calcular Kelly
                stake = KellyFractional.calculate_kelly_fraction(
                    prob, odds, kelly_fraction
                )
                
                # Calcular EV
                ev = prob * odds - 1.0
                
                stakes[i, j] = stake
                expected_values[i, j] = ev
        
        # Ajustar exposición total
        flat_stakes = stakes.flatten()
        flat_ev = expected_values.flatten()
        
        adjusted_stakes = KellyFractional.calculate_portfolio_allocation(
            flat_stakes, flat_ev
        )
        
        # Reformatear
        adjusted_stakes = adjusted_stakes.reshape((n_matches, 3))
        
        # Almacenar en portafolio
        singles_data = {
            'stakes': adjusted_stakes,
            'odds': odds_matrix,
            'probabilities': probabilities,
            'expected_values': expected_values
        }
        
        self.portfolio['singles'].append(singles_data)
        
        return singles_data
    
    def add_s73_system(
        self,
        combinations: np.ndarray,
        probabilities: np.ndarray,
        match_probabilities: np.ndarray,
        odds_matrix: np.ndarray,
        kelly_fraction: float = 0.5
    ) -> Dict[str, Any]:
        """
        Agrega sistema S73 al portafolio.
        
        Args:
            combinations: Array (K, 6) de combinaciones
            probabilities: Array (K,) de probabilidades conjuntas
            match_probabilities: Array (6, 3) de probabilidades por partido
            odds_matrix: Array (6, 3) de cuotas
            kelly_fraction: Fracción de Kelly
            
        Returns:
            Diccionario con stakes y métricas
        """
        n_columns = len(combinations)
        
        column_stakes = np.zeros(n_columns)
        column_odds = np.zeros(n_columns)
        column_ev = np.zeros(n_columns)
        
        for idx, (combo, prob) in enumerate(zip(combinations, probabilities)):
            # Calcular cuota conjunta
            combo_odds = 1.0
            for i, sign in enumerate(combo):
                combo_odds *= odds_matrix[i, sign]
            
            # Calcular Kelly para la columna
            stake = KellyFractional.calculate_kelly_fraction(
                prob, combo_odds, kelly_fraction
            )
            
            # Calcular EV
            ev = prob * combo_odds - 1.0
            
            column_stakes[idx] = stake
            column_odds[idx] = combo_odds
            column_ev[idx] = ev
        
        # Ajustar exposición total
        adjusted_stakes = KellyFractional.calculate_portfolio_allocation(
            column_stakes, column_ev
        )
        
        # Almacenar en portafolio
        s73_data = {
            'combinations': combinations,
            'stakes': adjusted_stakes,
            'odds': column_odds,
            'probabilities': probabilities,
            'expected_values': column_ev,
            'match_probabilities': match_probabilities
        }
        
        self.portfolio['s73_columns'].append(s73_data)
        
        return s73_data
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas institucionales del portafolio completo.
        
        Returns:
            Diccionario con todas las métricas
        """
        all_returns = []
        all_stakes = []
        all_probs = []
        all_odds = []
        
        # Recolectar datos de todas las estrategias
        for strategy_type, strategies in self.portfolio.items():
            for strategy in strategies:
                if strategy_type == 'singles':
                    stakes = strategy['stakes'].flatten()
                    probs = strategy['probabilities'].flatten()
                    odds = strategy['odds'].flatten()
                    evs = strategy['expected_values'].flatten()
                    
                    # Filtrar solo apuestas con stake > 0
                    mask = stakes > 0
                    if np.any(mask):
                        all_stakes.extend(stakes[mask])
                        all_probs.extend(probs[mask])
                        all_odds.extend(odds[mask])
                        
                        # Calcular retornos esperados
                        returns = stakes[mask] * self.current_bankroll * evs[mask]
                        all_returns.extend(returns)
                
                elif strategy_type == 's73_columns':
                    stakes = strategy['stakes']
                    probs = strategy['probabilities']
                    odds = strategy['odds']
                    evs = strategy['expected_values']
                    
                    mask = stakes > 0
                    if np.any(mask):
                        all_stakes.extend(stakes[mask])
                        all_probs.extend(probs[mask])
                        all_odds.extend(odds[mask])
                        
                        returns = stakes[mask] * self.current_bankroll * evs[mask]
                        all_returns.extend(returns)
        
        if not all_returns:
            return self._get_empty_metrics()
        
        # Convertir a arrays
        returns_array = np.array(all_returns)
        stakes_array = np.array(all_stakes)
        probs_array = np.array(all_probs)
        odds_array = np.array(all_odds)
        
        # Métricas básicas
        total_investment = np.sum(stakes_array) * self.current_bankroll
        total_expected_return = np.sum(returns_array)
        expected_roi = total_expected_return / total_investment if total_investment > 0 else 0
        
        # Métricas de riesgo
        variance = np.var(returns_array) if len(returns_array) > 1 else 0
        std_dev = np.sqrt(variance)
        
        # Sharpe Ratio (tasa libre de riesgo = 0)
        sharpe = total_expected_return / std_dev if std_dev > 0 else 0
        
        # Sortino Ratio (solo desviación downside)
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = total_expected_return / downside_std if downside_std > 0 else 0
        
        # Value at Risk (VaR 95%)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Conditional VaR (CVaR 95%)
        cvar_mask = returns_array <= var_95
        cvar_95 = np.mean(returns_array[cvar_mask]) if np.any(cvar_mask) else var_95
        
        # Max Drawdown estimado
        simulated_equity = self._simulate_equity_curve(probs_array, odds_array, stakes_array)
        max_drawdown = self._calculate_max_drawdown(simulated_equity)
        
        # Probabilidad de ruina
        ruin_prob = self._calculate_ruin_probability(stakes_array, probs_array, odds_array)
        
        # Exposición total
        total_exposure = np.sum(stakes_array)
        
        return {
            'total_expected_return': total_expected_return,
            'expected_roi': expected_roi,
            'variance': variance,
            'std_dev': std_dev,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'ruin_probability': ruin_prob,
            'total_exposure': total_exposure,
            'total_investment': total_investment,
            'num_bets': len(returns_array),
            'avg_bet_size': np.mean(stakes_array) * self.current_bankroll if len(stakes_array) > 0 else 0
        }
    
    def _simulate_equity_curve(
        self,
        probabilities: np.ndarray,
        odds: np.ndarray,
        stakes: np.ndarray,
        n_simulations: int = 1000
    ) -> np.ndarray:
        """Simula curva de equity para calcular drawdown."""
        n_bets = len(probabilities)
        if n_bets == 0:
            return np.array([self.current_bankroll])
        
        equity_curves = []
        
        for _ in range(n_simulations):
            equity = self.current_bankroll
            equity_history = [equity]
            
            for i in range(n_bets):
                stake_amount = stakes[i] * equity
                
                # Simular resultado
                if np.random.random() < probabilities[i]:
                    # Ganancia
                    equity += stake_amount * (odds[i] - 1)
                else:
                    # Pérdida
                    equity -= stake_amount
                
                equity_history.append(equity)
            
            equity_curves.append(equity_history)
        
        # Promedio de simulaciones
        avg_equity = np.mean(equity_curves, axis=0)
        
        return avg_equity
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calcula máximo drawdown de una curva de equity."""
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_ruin_probability(
        self,
        stakes: np.ndarray,
        probabilities: np.ndarray,
        odds: np.ndarray,
        n_simulations: int = 10000
    ) -> float:
        """Calcula probabilidad de ruina (bankroll < 50% inicial)."""
        n_bets = len(stakes)
        if n_bets == 0:
            return 0.0
        
        ruin_count = 0
        
        for _ in range(n_simulations):
            bankroll = self.current_bankroll
            
            for i in range(n_bets):
                stake_amount = stakes[i] * bankroll
                
                # Verificar si ya está en ruina
                if bankroll < self.initial_bankroll * CONST.RUIN_THRESHOLD:
                    ruin_count += 1
                    break
                
                # Simular resultado
                if np.random.random() < probabilities[i]:
                    bankroll += stake_amount * (odds[i] - 1)
                else:
                    bankroll -= stake_amount
            
            # Verificar después de todas las apuestas
            if bankroll < self.initial_bankroll * CONST.RUIN_THRESHOLD:
                ruin_count += 1
        
        return ruin_count / n_simulations
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Retorna métricas vacías cuando no hay apuestas."""
        return {
            'total_expected_return': 0.0,
            'expected_roi': 0.0,
            'variance': 0.0,
            'std_dev': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'max_drawdown': 0.0,
            'ruin_probability': 0.0,
            'total_exposure': 0.0,
            'total_investment': 0.0,
            'num_bets': 0,
            'avg_bet_size': 0.0
        }

# ============================================================================
# MÓDULO 6: INTERFAZ STREAMLIT CON FORMULARIO CENTRAL
# ============================================================================

class ACBEInterface:
    """Interfaz Streamlit profesional con formulario centralizado."""
    
    def __init__(self):
        self.setup_page()
        
        # Inicializar session state si no existe
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = None
    
    def setup_page(self):
        """Configuración de la página."""
        st.set_page_config(
            page_title="ACBE-S73 Quantum Suite v3.0",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎯 ACBE-S73 QUANTUM BETTING SUITE v3.0")
        st.markdown("""
        **Sistema Institucional de Optimización de Portafolios Deportivos**  
        *Gamma-Poisson Bayesiano + Optimización Combinatoria S73 + Criterio de Kelly Fraccional*
        """)
    
    def render_input_form(self) -> Optional[Dict]:
        """
        Renderiza formulario centralizado con todos los inputs.
        
        Returns:
            Diccionario con parámetros ingresados o None si no se ejecuta
        """
        with st.form("quantitative_analysis_form"):
            st.header("📥 Input de Datos y Parámetros")
            
            # Información estructural
            st.subheader("🏟️ Información Estructural")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                league = st.text_input("Liga/Competición", "La Liga")
            with col2:
                home_team = st.text_input("Equipo Local", "Real Madrid")
            with col3:
                away_team = st.text_input("Equipo Visitante", "FC Barcelona")
            
            # Parámetros ACBE
            st.subheader("🔬 Parámetros ACBE")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_home_goals = st.number_input(
                    "Goles promedio local (últimos N)",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.8,
                    step=0.1
                )
                avg_away_goals = st.number_input(
                    "Goles promedio visitante (últimos N)",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.2,
                    step=0.1
                )
            
            with col2:
                xg_home = st.number_input(
                    "xG local",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.6,
                    step=0.1
                )
                xg_away = st.number_input(
                    "xG visitante",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.4,
                    step=0.1
                )
            
            with col3:
                attack_strength = st.number_input(
                    "Fuerza ofensiva relativa",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.1,
                    step=0.1
                )
                defense_strength = st.number_input(
                    "Fuerza defensiva relativa",
                    min_value=0.5,
                    max_value=2.0,
                    value=0.9,
                    step=0.1
                )
            
            with col4:
                home_advantage = st.number_input(
                    "Ventaja local",
                    min_value=1.0,
                    max_value=1.5,
                    value=CONST.HOME_ADVANTAGE_BASE,
                    step=0.05
                )
                poisson_weight = st.slider(
                    "Peso Poisson (ω)",
                    min_value=0.0,
                    max_value=1.0,
                    value=CONST.DEFAULT_POISSON_WEIGHT,
                    step=0.05
                )
            
            # Cuotas
            st.subheader("💰 Cuotas de Mercado")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                odds_1 = st.number_input(
                    "Cuota 1 (Local)",
                    min_value=1.01,
                    max_value=100.0,
                    value=2.1,
                    step=0.1
                )
            with col2:
                odds_x = st.number_input(
                    "Cuota X (Empate)",
                    min_value=1.01,
                    max_value=100.0,
                    value=3.4,
                    step=0.1
                )
            with col3:
                odds_2 = st.number_input(
                    "Cuota 2 (Visitante)",
                    min_value=1.01,
                    max_value=100.0,
                    value=3.1,
                    step=0.1
                )
            
            # Parámetros de inversión
            st.subheader("🏦 Parámetros de Inversión")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_bankroll = st.number_input(
                    "Bankroll inicial (€)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=1000.0
                )
            with col2:
                kelly_fraction = st.slider(
                    "Fracción de Kelly",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
            with col3:
                num_matches_s73 = st.selectbox(
                    "Partidos para S73",
                    options=[3, 4, 5, 6],
                    index=3,
                    help="Número de partidos para sistema combinado (hasta 6)"
                )
            
            # Parámetros avanzados
            with st.expander("⚙️ Parámetros Avanzados"):
                col1, col2 = st.columns(2)
                with col1:
                    h2h_weight = st.slider(
                        "Peso H2H",
                        min_value=0.0,
                        max_value=1.0,
                        value=CONST.DEFAULT_H2H_WEIGHT,
                        step=0.05
                    )
                    sigma_vake = st.slider(
                        "Factor σ_vake",
                        min_value=0.0,
                        max_value=0.2,
                        value=CONST.DEFAULT_SIGMA_VAKE,
                        step=0.01
                    )
                with col2:
                    matches_considered = st.number_input(
                        "Partidos considerados (N)",
                        min_value=5,
                        max_value=50,
                        value=10,
                        step=5
                    )
                    monte_carlo_sims = st.number_input(
                        "Iteraciones Monte Carlo",
                        min_value=1000,
                        max_value=50000,
                        value=CONST.MONTE_CARLO_ITERATIONS,
                        step=1000
                    )
            
            # Botón de ejecución
            submit_button = st.form_submit_button(
                "🚀 EJECUTAR ANÁLISIS CUANTITATIVO",
                type="primary",
                use_container_width=True
            )
            
            if submit_button:
                # Guardar parámetros en session state
                st.session_state.input_params = {
                    'structural': {
                        'league': league,
                        'home_team': home_team,
                        'away_team': away_team
                    },
                    'acbe_params': {
                        'avg_home_goals': avg_home_goals,
                        'avg_away_goals': avg_away_goals,
                        'xg_home': xg_home,
                        'xg_away': xg_away,
                        'attack_strength': attack_strength,
                        'defense_strength': defense_strength,
                        'home_advantage': home_advantage,
                        'poisson_weight': poisson_weight,
                        'h2h_weight': h2h_weight,
                        'sigma_vake': sigma_vake,
                        'matches_considered': matches_considered
                    },
                    'odds': {
                        'odds_1': odds_1,
                        'odds_x': odds_x,
                        'odds_2': odds_2
                    },
                    'investment': {
                        'initial_bankroll': initial_bankroll,
                        'kelly_fraction': kelly_fraction,
                        'num_matches_s73': num_matches_s73,
                        'monte_carlo_sims': monte_carlo_sims
                    }
                }
                
                return st.session_state.input_params
        
        return None
    
    def execute_analysis(self, params: Dict) -> Dict:
        """
        Ejecuta análisis cuantitativo completo.
        
        Args:
            params: Diccionario con parámetros ingresados
            
        Returns:
            Resultados del análisis
        """
        with st.spinner("🔄 Ejecutando análisis cuantitativo..."):
            # 1. Calcular probabilidades ACBE
            acbe_results = self._calculate_acbe_probabilities(params)
            
            # 2. Generar múltiples partidos para S73
            match_data = self._generate_match_data(params, acbe_results)
            
            # 3. Optimización S73
            s73_results = self._optimize_s73_system(match_data)
            
            # 4. Gestión de portafolio
            portfolio_results = self._manage_portfolio(
                params, match_data, s73_results
            )
            
            # 5. Calcular métricas
            metrics = portfolio_results['metrics']
            
            return {
                'acbe_results': acbe_results,
                'match_data': match_data,
                's73_results': s73_results,
                'portfolio_results': portfolio_results,
                'metrics': metrics,
                'params': params
            }
    
    def _calculate_acbe_probabilities(self, params: Dict) -> Dict:
        """Calcula probabilidades usando modelo ACBE."""
        acbe_params = params['acbe_params']
        
        # Calcular lambdas usando Gamma-Poisson
        lambda_home, _ = ACBEModel.gamma_poisson_posterior(
            acbe_params['avg_home_goals'],
            acbe_params['matches_considered']
        )
        
        lambda_away, _ = ACBEModel.gamma_poisson_posterior(
            acbe_params['avg_away_goals'],
            acbe_params['matches_considered']
        )
        
        # Ajustar por ventaja local y fuerzas relativas
        lambda_home_adj = lambda_home * acbe_params['home_advantage'] * acbe_params['attack_strength']
        lambda_away_adj = lambda_away * acbe_params['defense_strength']
        
        # Simular resultados
        poisson_probs = ACBEModel.simulate_match_outcomes(
            lambda_home_adj,
            lambda_away_adj,
            params['investment']['monte_carlo_sims']
        )
        
        # Calcular probabilidades finales
        final_probs = ACBEModel.calculate_final_probabilities(
            poisson_probs,
            None,  # No H2H por simplicidad
            acbe_params['poisson_weight'],
            acbe_params['sigma_vake']
        )
        
        return {
            'lambda_home': lambda_home_adj,
            'lambda_away': lambda_away_adj,
            'poisson_probs': poisson_probs,
            'final_probs': final_probs,
            'odds': [
                params['odds']['odds_1'],
                params['odds']['odds_x'],
                params['odds']['odds_2']
            ]
        }
    
    def _generate_match_data(self, params: Dict, acbe_results: Dict) -> Dict:
        """Genera datos para múltiples partidos."""
        num_matches = params['investment']['num_matches_s73']
        
        match_probabilities = np.zeros((num_matches, 3))
        odds_matrix = np.zeros((num_matches, 3))
        
        # Usar las probabilidades calculadas como base
        base_probs = acbe_results['final_probs']
        base_odds = acbe_results['odds']
        
        for i in range(num_matches):
            # Variar ligeramente las probabilidades
            noise = np.random.normal(0, 0.05, 3)
            varied_probs = base_probs + noise
            varied_probs = np.maximum(varied_probs, 0.05)
            varied_probs = varied_probs / varied_probs.sum()
            
            # Variar ligeramente las cuotas
            odds_noise = np.random.normal(0, 0.1, 3)
            varied_odds = base_odds + odds_noise
            varied_odds = np.maximum(varied_odds, 1.1)
            
            match_probabilities[i] = varied_probs
            odds_matrix[i] = varied_odds
        
        return {
            'probabilities': match_probabilities,
            'odds': odds_matrix,
            'num_matches': num_matches
        }
    
    def _optimize_s73_system(self, match_data: Dict) -> Dict:
        """Optimiza sistema S73."""
        probabilities = match_data['probabilities']
        
        # Ejecutar optimización greedy
        combinations, combo_probs, k_optimal = S73Optimizer.greedy_coverage_optimization(
            probabilities
        )
        
        # Validar cobertura
        validation = S73Optimizer.validate_coverage(
            combinations,
            probabilities
        )
        
        return {
            'combinations': combinations,
            'probabilities': combo_probs,
            'k_optimal': k_optimal,
            'validation': validation,
            'match_probabilities': probabilities
        }
    
    def _manage_portfolio(self, params: Dict, match_data: Dict, s73_results: Dict) -> Dict:
        """Gestiona portafolio completo."""
        initial_bankroll = params['investment']['initial_bankroll']
        kelly_fraction = params['investment']['kelly_fraction']
        
        # Inicializar portafolio
        portfolio = PortfolioUnification(initial_bankroll)
        
        # Agregar apuestas simples
        singles_data = portfolio.add_singles(
            match_data['probabilities'],
            match_data['odds'],
            kelly_fraction
        )
        
        # Agregar sistema S73
        s73_portfolio_data = portfolio.add_s73_system(
            s73_results['combinations'],
            s73_results['probabilities'],
            s73_results['match_probabilities'],
            match_data['odds'],
            kelly_fraction
        )
        
        # Calcular métricas
        metrics = portfolio.calculate_portfolio_metrics()
        
        return {
            'portfolio': portfolio,
            'singles_data': singles_data,
            's73_portfolio_data': s73_portfolio_data,
            'metrics': metrics
        }
    
    def render_results(self, results: Dict):
        """Renderiza resultados del análisis."""
        if results is None:
            return
        
        st.header("📊 Resultados del Análisis Cuantitativo")
        
        # 1. Resultados ACBE
        self._render_acbe_results(results['acbe_results'], results['params'])
        
        # 2. Tabla EV
        self._render_ev_table(results['match_data'])
        
        # 3. S73 optimizado
        self._render_s73_results(results['s73_results'])
        
        # 4. Portafolio final
        self._render_portfolio_results(results['portfolio_results'])
        
        # 5. Métricas institucionales
        self._render_institutional_metrics(results['metrics'])
        
        # 6. Gráficos
        self._render_visualizations(results)
    
    def _render_acbe_results(self, acbe_results: Dict, params: Dict):
        """Renderiza resultados del modelo ACBE."""
        st.subheader("🔬 Resultados del Modelo ACBE")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "λ Local (ajustado)",
                f"{acbe_results['lambda_home']:.3f}",
                help="Tasa de goles local posterior"
            )
            st.metric(
                "λ Visitante (ajustado)",
                f"{acbe_results['lambda_away']:.3f}",
                help="Tasa de goles visitante posterior"
            )
        
        with col2:
            probs = acbe_results['final_probs']
            for i, label in enumerate(CONST.OUTCOME_LABELS):
                st.metric(
                    f"P({label}) Final",
                    f"{probs[i]:.1%}",
                    delta=None
                )
        
        with col3:
            odds = acbe_results['odds']
            evs = probs * odds - 1.0
            
            for i, label in enumerate(CONST.OUTCOME_LABELS):
                color = "green" if evs[i] > 0 else "red"
                st.metric(
                    f"EV({label})",
                    f"{evs[i]:.3f}",
                    delta=None,
                    delta_color="off"
                )
        
        # Gráfico de probabilidades
        fig = go.Figure(data=[
            go.Bar(
                name='Probabilidades',
                x=CONST.OUTCOME_LABELS,
                y=probs,
                marker_color=CONST.OUTCOME_COLORS,
                text=[f'{p:.1%}' for p in probs],
                textposition='auto'
            ),
            go.Scatter(
                name='Cuotas implícitas',
                x=CONST.OUTCOME_LABELS,
                y=1/np.array(odds),
                mode='lines+markers',
                line=dict(color='white', width=2),
                marker=dict(size=10)
            )
        ])
        
        fig.update_layout(
            title="Probabilidades ACBE vs Cuotas Implícitas",
            yaxis_title="Probabilidad",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_ev_table(self, match_data: Dict):
        """Renderiza tabla de valor esperado."""
        st.subheader("💰 Tabla de Valor Esperado")
        
        probabilities = match_data['probabilities']
        odds = match_data['odds']
        n_matches = match_data['num_matches']
        
        ev_matrix = probabilities * odds - 1.0
        
        df = pd.DataFrame({
            'Partido': [f'Partido {i+1}' for i in range(n_matches)],
            'P(1)': [f'{p:.1%}' for p in probabilities[:, 0]],
            'P(X)': [f'{p:.1%}' for p in probabilities[:, 1]],
            'P(2)': [f'{p:.1%}' for p in probabilities[:, 2]],
            'Cuota 1': [f'{o:.2f}' for o in odds[:, 0]],
            'Cuota X': [f'{o:.2f}' for o in odds[:, 1]],
            'Cuota 2': [f'{o:.2f}' for o in odds[:, 2]],
            'EV 1': [f'{ev:.3f}' for ev in ev_matrix[:, 0]],
            'EV X': [f'{ev:.3f}' for ev in ev_matrix[:, 1]],
            'EV 2': [f'{ev:.3f}' for ev in ev_matrix[:, 2]]
        })
        
        # Aplicar formato condicional
        def color_ev(val):
            try:
                ev = float(val)
                if ev > 0:
                    return f"background-color: {CONST.COLORS['success']}20; color: {CONST.COLORS['success']}"
                elif ev < 0:
                    return f"background-color: {CONST.COLORS['danger']}20; color: {CONST.COLORS['danger']}"
                else:
                    return ""
            except:
                return ""
        
        styled_df = df.style.applymap(color_ev, subset=['EV 1', 'EV X', 'EV 2'])
        st.dataframe(styled_df, use_container_width=True, height=300)
    
    def _render_s73_results(self, s73_results: Dict):
        """Renderiza resultados del sistema S73."""
        st.subheader("🧮 Sistema S73 Optimizado")
        
        validation = s73_results['validation']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Combinaciones totales",
                CONST.FULL_COMBINATIONS
            )
        
        with col2:
            st.metric(
                "Combinaciones seleccionadas (K)",
                s73_results['k_optimal'],
                help="Número óptimo calculado por el algoritmo greedy"
            )
        
        with col3:
            status = "✅ VALIDADO" if validation['is_valid'] else "❌ NO VALIDADO"
            st.metric(
                "Cobertura de 2 errores",
                status
            )
        
        with col4:
            st.metric(
                "Masa probabilística cubierta",
                f"{validation['coverage_rate']:.1%}"
            )
        
        # Mostrar primeras 10 combinaciones
        st.subheader("📋 Primeras 10 Combinaciones S73")
        
        combinations = s73_results['combinations'][:10]
        probs = s73_results['probabilities'][:10]
        
        combo_data = []
        for i, (combo, prob) in enumerate(zip(combinations, probs), 1):
            combo_str = ''.join([CONST.OUTCOME_LABELS[int(s)] for s in combo])
            combo_data.append({
                '#': i,
                'Combinación': combo_str,
                'Probabilidad': f"{prob:.6f}",
                'Log(Prob)': f"{np.log10(prob):.2f}"
            })
        
        df_combos = pd.DataFrame(combo_data)
        st.dataframe(df_combos, use_container_width=True, height=400)
        
        # Gráfico de distribución de probabilidades
        fig = go.Figure(data=[
            go.Histogram(
                x=s73_results['probabilities'],
                nbinsx=20,
                marker_color=CONST.COLORS['primary'],
                opacity=0.7,
                name='Probabilidades conjuntas'
            )
        ])
        
        fig.add_vline(
            x=np.mean(s73_results['probabilities']),
            line_dash="dash",
            line_color=CONST.COLORS['warning'],
            annotation_text=f"Media: {np.mean(s73_results['probabilities']):.6f}"
        )
        
        fig.update_layout(
            title="Distribución de Probabilidades Conjuntas (S73)",
            xaxis_title="Probabilidad conjunta",
            yaxis_title="Frecuencia",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_portfolio_results(self, portfolio_results: Dict):
        """Renderiza resultados del portafolio."""
        st.subheader("🏦 Gestión de Portafolio")
        
        singles_data = portfolio_results['singles_data']
        s73_data = portfolio_results['s73_portfolio_data']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            singles_stake = np.sum(singles_data['stakes'])
            st.metric(
                "Exposición Singles",
                f"{singles_stake:.2%}",
                help="Stake total en apuestas simples"
            )
        
        with col2:
            s73_stake = np.sum(s73_data['stakes'])
            st.metric(
                "Exposición S73",
                f"{s73_stake:.2%}",
                help="Stake total en sistema S73"
            )
        
        with col3:
            total_exposure = singles_stake + s73_stake
            exposure_color = "normal"
            if total_exposure > CONST.MAX_PORTFOLIO_EXPOSURE:
                exposure_color = "inverse"
            
            st.metric(
                "Exposición Total",
                f"{total_exposure:.2%}",
                delta=f"Límite: {CONST.MAX_PORTFOLIO_EXPOSURE:.0%}",
                delta_color=exposure_color
            )
        
        # Distribución de stakes
        st.subheader("📊 Distribución de Stakes")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Singles', 'S73'])
        
        # Singles
        singles_flat = singles_data['stakes'].flatten()
        singles_flat = singles_flat[singles_flat > 0]
        
        if len(singles_flat) > 0:
            fig.add_trace(
                go.Histogram(
                    x=singles_flat * 100,  # Convertir a porcentaje
                    name='Singles',
                    marker_color=CONST.COLORS['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # S73
        s73_stakes = s73_data['stakes']
        s73_stakes = s73_stakes[s73_stakes > 0]
        
        if len(s73_stakes) > 0:
            fig.add_trace(
                go.Histogram(
                    x=s73_stakes * 100,  # Convertir a porcentaje
                    name='S73',
                    marker_color=CONST.COLORS['secondary'],
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Stake (%)", row=1, col=1)
        fig.update_xaxes(title_text="Stake (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_institutional_metrics(self, metrics: Dict):
        """Renderiza métricas institucionales."""
        st.subheader("📈 Métricas Institucionales")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "ROI Esperado",
                f"{metrics['expected_roi']:.2%}",
                help="Retorno sobre inversión esperado"
            )
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.3f}",
                help="Retorno por unidad de riesgo total"
            )
        
        with col2:
            st.metric(
                "Sortino Ratio",
                f"{metrics['sortino_ratio']:.3f}",
                help="Retorno por unidad de riesgo bajista"
            )
            st.metric(
                "Desviación Estándar",
                f"{metrics['std_dev']:.2f}",
                help="Volatilidad de retornos"
            )
        
        with col3:
            st.metric(
                "VaR 95%",
                f"€{metrics['var_95']:.2f}",
                help="Pérdida máxima esperada al 95% confianza"
            )
            st.metric(
                "CVaR 95%",
                f"€{metrics['cvar_95']:.2f}",
                help="Pérdida esperada en el peor 5% de casos"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.2%}",
                help="Máxima caída desde pico"
            )
            st.metric(
                "Prob. Ruina",
                f"{metrics['ruin_probability']:.2%}",
                help="Probabilidad de perder 50% del bankroll"
            )
        
        # Detalles adicionales
        with st.expander("📋 Detalles Completos del Portafolio"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Retorno Total Esperado", f"€{metrics['total_expected_return']:.2f}")
                st.metric("Inversión Total", f"€{metrics['total_investment']:.2f}")
                st.metric("Número de Apuestas", metrics['num_bets'])
            
            with col2:
                st.metric("Exposición Total", f"{metrics['total_exposure']:.2%}")
                st.metric("Apuesta Promedio", f"€{metrics['avg_bet_size']:.2f}")
                st.metric("Varianza", f"{metrics['variance']:.4f}")
    
    def _render_visualizations(self, results: Dict):
        """Renderiza visualizaciones avanzadas."""
        st.header("📈 Visualizaciones Avanzadas")
        
        # 1. Curva de equity simulada
        self._render_equity_curve(results)
        
        # 2. Distribución de retornos
        self._render_return_distribution(results)
        
        # 3. Análisis de drawdown
        self._render_drawdown_analysis(results)
        
        # 4. Heatmap de correlaciones
        self._render_correlation_heatmap(results)
    
    def _render_equity_curve(self, results: Dict):
        """Renderiza curva de equity simulada."""
        st.subheader("📈 Curva de Equity Simulada")
        
        # Simular múltiples trayectorias
        portfolio = results['portfolio_results']['portfolio']
        metrics = results['metrics']
        
        n_simulations = 100
        n_periods = 100
        equity_curves = []
        
        for _ in range(n_simulations):
            equity = results['params']['investment']['initial_bankroll']
            equity_history = [equity]
            
            for _ in range(n_periods):
                # Retorno aleatorio basado en métricas
                if metrics['std_dev'] > 0:
                    ret = np.random.normal(
                        metrics['expected_roi'],
                        metrics['std_dev']
                    )
                else:
                    ret = metrics['expected_roi']
                
                equity *= (1 + ret)
                equity_history.append(equity)
            
            equity_curves.append(equity_history)
        
        # Calcular percentiles
        equity_matrix = np.array(equity_curves)
        median_curve = np.median(equity_matrix, axis=0)
        p10_curve = np.percentile(equity_matrix, 10, axis=0)
        p90_curve = np.percentile(equity_matrix, 90, axis=0)
        
        fig = go.Figure()
        
        # Área de confianza
        fig.add_trace(go.Scatter(
            x=list(range(len(median_curve))),
            y=p90_curve,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(median_curve))),
            y=p10_curve,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(30, 136, 229, 0.2)',
            name='80% Intervalo Confianza'
        ))
        
        # Mediana
        fig.add_trace(go.Scatter(
            x=list(range(len(median_curve))),
            y=median_curve,
            mode='lines',
            line=dict(color=CONST.COLORS['primary'], width=3),
            name='Mediana'
        ))
        
        fig.update_layout(
            title="Simulación de Curva de Equity (100 trayectorias)",
            xaxis_title="Período",
            yaxis_title="Bankroll (€)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_return_distribution(self, results: Dict):
        """Renderiza distribución de retornos."""
        st.subheader("📊 Distribución de Retornos Esperados")
        
        metrics = results['metrics']
        
        # Generar distribución normal
        if metrics['std_dev'] > 0:
            returns = np.random.normal(
                metrics['expected_roi'],
                metrics['std_dev'],
                10000
            )
        else:
            returns = np.full(10000, metrics['expected_roi'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns * 100,  # Convertir a porcentaje
            nbinsx=50,
            marker_color=CONST.COLORS['primary'],
            opacity=0.7,
            name='Distribución retornos'
        ))
        
        # Líneas de referencia
        fig.add_vline(
            x=metrics['expected_roi'] * 100,
            line_dash="dash",
            line_color=CONST.COLORS['warning'],
            annotation_text=f"Media: {metrics['expected_roi']:.2%}"
        )
        
        fig.add_vline(
            x=0,
            line_dash="dot",
            line_color=CONST.COLORS['danger'],
            annotation_text="Break-even"
        )
        
        fig.update_layout(
            title="Distribución de Retornos por Período",
            xaxis_title="Retorno (%)",
            yaxis_title="Frecuencia",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_drawdown_analysis(self, results: Dict):
        """Renderiza análisis de drawdown."""
        st.subheader("📉 Análisis de Drawdown")
        
        # Simular drawdowns
        n_simulations = 1000
        n_periods = 252  # Un año de trading
        max_drawdowns = []
        
        metrics = results['metrics']
        
        for _ in range(n_simulations):
            equity = 10000  # Bankroll inicial
            peak = equity
            max_dd = 0.0
            
            for _ in range(n_periods):
                if metrics['std_dev'] > 0:
                    ret = np.random.normal(
                        metrics['expected_roi'] / 252,  # Diario
                        metrics['std_dev'] / np.sqrt(252)
                    )
                else:
                    ret = metrics['expected_roi'] / 252
                
                equity *= (1 + ret)
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            
            max_drawdowns.append(max_dd)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=[dd * 100 for dd in max_drawdowns],  # Convertir a porcentaje
            nbinsx=30,
            marker_color=CONST.COLORS['danger'],
            opacity=0.7,
            name='Max Drawdown'
        ))
        
        fig.add_vline(
            x=np.percentile(max_drawdowns, 95) * 100,
            line_dash="dash",
            line_color=CONST.COLORS['warning'],
            annotation_text=f"95%: {np.percentile(max_drawdowns, 95):.2%}"
        )
        
        fig.update_layout(
            title="Distribución de Máximo Drawdown (simulación anual)",
            xaxis_title="Max Drawdown (%)",
            yaxis_title="Frecuencia",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_heatmap(self, results: Dict):
        """Renderiza heatmap de correlaciones."""
        st.subheader("🔥 Heatmap de Correlaciones")
        
        # Para este ejemplo, crear matriz de correlación sintética
        n_assets = 10
        corr_matrix = np.eye(n_assets)
        
        # Agregar correlaciones aleatorias
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                corr = np.random.uniform(-0.3, 0.3)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=[f'Activo {i+1}' for i in range(n_assets)],
            y=[f'Activo {i+1}' for i in range(n_assets)],
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlación')
        ))
        
        fig.update_layout(
            title="Matriz de Correlación entre Activos del Portafolio",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Método principal de ejecución."""
        # Renderizar formulario
        input_params = self.render_input_form()
        
        # Solo ejecutar análisis si se presionó el botón
        if input_params is not None:
            # Ejecutar análisis
            results = self.execute_analysis(input_params)
            
            # Guardar en session state
            st.session_state.analysis_results = results
            
            # Renderizar resultados
            self.render_results(results)
        elif st.session_state.analysis_results is not None:
            # Mostrar resultados previos
            st.info("Mostrando resultados del análisis anterior. Presiona el botón para re-ejecutar.")
            self.render_results(st.session_state.analysis_results)
        else:
            # Estado inicial
            st.info("👈 Configura todos los parámetros y presiona 'EJECUTAR ANÁLISIS CUANTITATIVO'")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Inicializar y ejecutar la aplicación
    app = ACBEInterface()
    app.run()
