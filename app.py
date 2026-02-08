"""
🎯 ACBE QUANTUM BETTING SUITE v3.0
Sistema cuantitativo institucional para optimización de portafolios de apuestas deportivas
Arquitectura unificada: Singles + Combinadas + Sistema S73 Optimizado
Con modelo de penalización por entropía, anti-sobreoptimización y gestión de riesgo avanzada
Autor: Arquitecto de Software & Data Scientist Senior (Quantitative Betting)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import itertools
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import math
warnings.filterwarnings('ignore')

# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL SISTEMA Y CONSTANTES MATEMÁTICAS
# ============================================================================

class SystemConfig:
    """Configuración centralizada del sistema ACBE v3.0."""
    
    # Parámetros fundamentales
    BASE_ENTROPY = 3  # Base logarítmica para 3 resultados
    MIN_PROBABILITY = 1e-10
    LOG_BASE_3 = np.log(3)
    
    # Sistema S73
    NUM_MATCHES = 6
    FULL_COMBINATIONS = 3 ** 6
    HAMMING_DISTANCE_TARGET = 2
    
    # Hiperparámetros del modelo (ajustables por usuario)
    DEFAULT_LAMBDA = 0.3     # Penalización por entropía
    DEFAULT_MU = 0.2        # Penalización por drawdown
    DEFAULT_ALPHA = 0.5     # Factor anti-sobreoptimización
    DEFAULT_RHO = 0.4       # Factor conservador Kelly
    
    # Límites de gestión
    MIN_ODDS = 1.01
    MAX_ODDS = 100.0
    DEFAULT_BANKROLL = 10000.0
    MAX_KELLY_FRACTION = 0.05  # 5% máximo por apuesta
    MAX_PORTFOLIO_EXPOSURE = 0.20  # 20% máximo del bankroll
    
    # Umbrales de decisión
    EV_THRESHOLD_SINGLE = 0.05     # EV mínimo para single
    EV_THRESHOLD_COMBINED = 0.10   # EV mínimo para combinada
    ENTROPY_THRESHOLD = 0.7        # Entropía máxima aceptable
    
    # Configuración visual
    COLORS = {
        'primary': '#1E88E5',
        'secondary': '#FFC107', 
        'success': '#4CAF50',
        'danger': '#F44336',
        'warning': '#FF9800',
        'info': '#00BCD4'
    }
    
    # Mapeo de resultados
    OUTCOME_MAPPING = {'1': 0, 'X': 1, '2': 2}
    OUTCOME_LABELS = ['1', 'X', '2']
    OUTCOME_COLORS = ['#1E88E5', '#FF9800', '#F44336']

# ============================================================================
# SECCIÓN 2: MODELO MATEMÁTICO AVANZADO (CORE CUANTITATIVO)
# ============================================================================

@dataclass
class MatchData:
    """Estructura de datos para partido individual."""
    match_id: int
    league: str
    home_team: str
    away_team: str
    odds_1: float
    odds_X: float
    odds_2: float
    prob_1: float
    prob_X: float
    prob_2: float
    entropy: float = 0.0
    
    def __post_init__(self):
        """Calcula entropía del partido después de inicialización."""
        probs = np.array([self.prob_1, self.prob_X, self.prob_2])
        probs = np.clip(probs, SystemConfig.MIN_PROBABILITY, 1.0)
        self.entropy = -np.sum(probs * np.log(probs) / SystemConfig.LOG_BASE_3)
    
    @property
    def odds_matrix(self) -> np.ndarray:
        return np.array([self.odds_1, self.odds_X, self.odds_2])
    
    @property
    def prob_matrix(self) -> np.ndarray:
        return np.array([self.prob_1, self.prob_X, self.prob_2])
    
    @property
    def ev_matrix(self) -> np.ndarray:
        return self.prob_matrix * self.odds_matrix - 1

class QuantitativeModel:
    """Modelo matemático avanzado con penalizaciones cuantitativas."""
    
    @staticmethod
    def calculate_column_entropy(column_signs: np.ndarray, 
                                match_probabilities: List[np.ndarray]) -> float:
        """
        Calcula entropía REAL de una columna (no promedio).
        
        Fórmula: H_j = (1/6) * Σ_i [-log_3(P_i,signo_i,j)]
        
        Args:
            column_signs: Array (6,) con signos de la columna (0,1,2)
            match_probabilities: Lista de arrays (3,) con probabilidades por partido
            
        Returns:
            Entropía de la columna (0-1)
        """
        total_entropy = 0.0
        
        for i, sign in enumerate(column_signs):
            prob = match_probabilities[i][sign]
            prob = max(prob, SystemConfig.MIN_PROBABILITY)
            total_entropy += -np.log(prob) / SystemConfig.LOG_BASE_3
        
        return total_entropy / 6.0
    
    @staticmethod
    def calculate_adjusted_joint_probability(column_signs: np.ndarray,
                                           match_probabilities: List[np.ndarray],
                                           column_entropy: float,
                                           alpha: float = SystemConfig.DEFAULT_ALPHA) -> float:
        """
        Calcula probabilidad conjunta ajustada con penalización anti-sobreoptimización.
        
        Fórmula: P_joint,adj = (Π_i P_i,signo_i,j) * exp(-α * H_j)
        
        Args:
            column_signs: Array (6,) con signos
            match_probabilities: Lista de arrays (3,) con probabilidades
            column_entropy: Entropía de la columna
            alpha: Factor anti-sobreoptimización (0.3-0.7)
            
        Returns:
            Probabilidad conjunta ajustada
        """
        # Probabilidad conjunta cruda
        raw_prob = 1.0
        for i, sign in enumerate(column_signs):
            raw_prob *= match_probabilities[i][sign]
        
        # Ajuste anti-sobreoptimización
        adjusted_prob = raw_prob * np.exp(-alpha * column_entropy)
        
        return max(adjusted_prob, SystemConfig.MIN_PROBABILITY)
    
    @staticmethod
    def calculate_column_ev(adjusted_prob: float, 
                           column_odds: float) -> float:
        """
        Calcula Expected Value de una columna.
        
        Fórmula: EV_j = P_joint,adj * Odds_j - 1
        
        Args:
            adjusted_prob: Probabilidad conjunta ajustada
            column_odds: Cuota conjunta de la columna
            
        Returns:
            Expected Value
        """
        return adjusted_prob * column_odds - 1
    
    @staticmethod
    def estimate_drawdown_sensitivity(column_odds: float, 
                                     column_entropy: float) -> float:
        """
        Estimación proxy de sensibilidad al drawdown.
        
        Basado en odds (alto = riesgo) y entropía (alta = incertidumbre).
        
        Args:
            column_odds: Cuota conjunta
            column_entropy: Entropía de la columna
            
        Returns:
            Sensibilidad al drawdown (0-1)
        """
        # Normalizar odds (log scale para capturar escala)
        odds_factor = np.log10(column_odds) / np.log10(100)
        odds_factor = np.clip(odds_factor, 0, 1)
        
        # Combinar factores
        dd_sensitivity = 0.6 * odds_factor + 0.4 * column_entropy
        
        return np.clip(dd_sensitivity, 0, 1)
    
    @staticmethod
    def calculate_column_score(ev: float,
                              entropy: float,
                              dd_sensitivity: float,
                              lambda_param: float = SystemConfig.DEFAULT_LAMBDA,
                              mu_param: float = SystemConfig.DEFAULT_MU) -> float:
        """
        Calcula score cuantitativo de una columna.
        
        Fórmula: Score_j = EV_j - λ*H_j - μ*DD_j
        
        Args:
            ev: Expected Value
            entropy: Entropía de la columna
            dd_sensitivity: Sensibilidad al drawdown
            lambda_param: Penalización por entropía
            mu_param: Penalización por drawdown
            
        Returns:
            Score de la columna
        """
        return ev - lambda_param * entropy - mu_param * dd_sensitivity

# ============================================================================
# SECCIÓN 3: MOTOR DE EVALUACIÓN (EVALUATION LAYER)
# ============================================================================

class BetType(Enum):
    """Tipos de apuesta disponibles."""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    SYSTEM_S73 = "system_s73"

@dataclass
class BetEvaluation:
    """Evaluación completa de una apuesta."""
    bet_type: BetType
    match_indices: List[int]
    signs: List[int]
    odds: float
    probability: float
    adjusted_probability: float
    entropy: float
    expected_value: float
    dd_sensitivity: float
    score: float
    kelly_fraction: float = 0.0
    
    @property
    def description(self) -> str:
        """Descripción legible de la apuesta."""
        if self.bet_type == BetType.SINGLE:
            return f"Single: Partido {self.match_indices[0] + 1}, Signo {SystemConfig.OUTCOME_LABELS[self.signs[0]]}"
        elif self.bet_type in [BetType.DOUBLE, BetType.TRIPLE]:
            matches = [f"{i+1}{SystemConfig.OUTCOME_LABELS[s]}" for i, s in zip(self.match_indices, self.signs)]
            return f"{self.bet_type.value.capitalize()}: {'-'.join(matches)}"
        else:
            return f"S73: {'-'.join([SystemConfig.OUTCOME_LABELS[s] for s in self.signs])}"

class EvaluationLayer:
    """Capa de evaluación que calcula métricas para todas las apuestas posibles."""
    
    def __init__(self, matches_data: List[MatchData]):
        self.matches = matches_data
        self.match_probs = [match.prob_matrix for match in matches_data]
        self.match_odds = [match.odds_matrix for match in matches_data]
        
    def evaluate_all_bets(self, 
                         max_combinations: int = 3,
                         alpha: float = SystemConfig.DEFAULT_ALPHA,
                         lambda_param: float = SystemConfig.DEFAULT_LAMBDA,
                         mu_param: float = SystemConfig.DEFAULT_MU) -> List[BetEvaluation]:
        """
        Evalúa todas las apuestas posibles: singles, dobles, triples.
        
        Args:
            max_combinations: Máximo de partidos por combinada (1=single, 2=dobles, 3=triples)
            alpha: Factor anti-sobreoptimización
            lambda_param, mu_param: Parámetros de penalización
            
        Returns:
            Lista de evaluaciones de apuestas
        """
        evaluations = []
        
        # Evaluar singles
        evaluations.extend(self._evaluate_singles(alpha, lambda_param, mu_param))
        
        # Evaluar combinadas (2 y 3 partidos)
        if max_combinations >= 2:
            evaluations.extend(self._evaluate_combined(2, alpha, lambda_param, mu_param))
        if max_combinations >= 3:
            evaluations.extend(self._evaluate_combined(3, alpha, lambda_param, mu_param))
        
        return evaluations
    
    def _evaluate_singles(self, alpha: float, lambda_param: float, mu_param: float) -> List[BetEvaluation]:
        """Evalúa todas las apuestas simples."""
        evaluations = []
        
        for match_idx, match in enumerate(self.matches):
            for sign_idx in range(3):
                prob = match.prob_matrix[sign_idx]
                odds = match.odds_matrix[sign_idx]
                ev = match.ev_matrix[sign_idx]
                
                # Entropía del "single" (es solo la incertidumbre de ese signo)
                # Para single, usamos 1 - probabilidad como proxy de entropía
                single_entropy = 1.0 - prob
                
                # Ajuste de probabilidad (aunque para single es menos crítico)
                adj_prob = prob * np.exp(-alpha * single_entropy)
                
                # Sensibilidad al drawdown (más alta para odds altas)
                dd_sens = QuantitativeModel.estimate_drawdown_sensitivity(odds, single_entropy)
                
                # Score
                score = QuantitativeModel.calculate_column_score(
                    ev, single_entropy, dd_sens, lambda_param, mu_param
                )
                
                eval_bet = BetEvaluation(
                    bet_type=BetType.SINGLE,
                    match_indices=[match_idx],
                    signs=[sign_idx],
                    odds=odds,
                    probability=prob,
                    adjusted_probability=adj_prob,
                    entropy=single_entropy,
                    expected_value=ev,
                    dd_sensitivity=dd_sens,
                    score=score
                )
                
                evaluations.append(eval_bet)
        
        return evaluations
    
    def _evaluate_combined(self, 
                          n_matches: int,
                          alpha: float,
                          lambda_param: float,
                          mu_param: float) -> List[BetEvaluation]:
        """Evalúa combinadas de n partidos."""
        evaluations = []
        n_total = len(self.matches)
        
        # Generar todas las combinaciones de partidos
        match_combinations = list(itertools.combinations(range(n_total), n_matches))
        
        for match_indices in match_combinations:
            # Para cada combinación de partidos, evaluar todas las combinaciones de signos
            sign_combinations = list(itertools.product(range(3), repeat=n_matches))
            
            for signs in sign_combinations:
                # Calcular probabilidad conjunta cruda
                joint_prob = 1.0
                joint_odds = 1.0
                
                for i, match_idx in enumerate(match_indices):
                    sign = signs[i]
                    joint_prob *= self.match_probs[match_idx][sign]
                    joint_odds *= self.match_odds[match_idx][sign]
                
                # Calcular entropía de la combinada
                # Para combinada, calculamos entropía promedio de los partidos
                combined_entropy = 0.0
                for i, match_idx in enumerate(match_indices):
                    combined_entropy += self.matches[match_idx].entropy
                combined_entropy /= n_matches
                
                # Probabilidad ajustada
                adj_prob = joint_prob * np.exp(-alpha * combined_entropy)
                
                # EV
                ev = adj_prob * joint_odds - 1
                
                # Sensibilidad al drawdown
                dd_sens = QuantitativeModel.estimate_drawdown_sensitivity(joint_odds, combined_entropy)
                
                # Score
                score = QuantitativeModel.calculate_column_score(
                    ev, combined_entropy, dd_sens, lambda_param, mu_param
                )
                
                # Determinar tipo de combinada
                bet_type = BetType.DOUBLE if n_matches == 2 else BetType.TRIPLE
                
                eval_bet = BetEvaluation(
                    bet_type=bet_type,
                    match_indices=list(match_indices),
                    signs=list(signs),
                    odds=joint_odds,
                    probability=joint_prob,
                    adjusted_probability=adj_prob,
                    entropy=combined_entropy,
                    expected_value=ev,
                    dd_sensitivity=dd_sens,
                    score=score
                )
                
                evaluations.append(eval_bet)
        
        return evaluations

# ============================================================================
# SECCIÓN 4: SISTEMA S73 ÓPTIMO (REDISEÑADO)
# ============================================================================

class OptimalS73System:
    """Sistema S73 óptimo con algoritmo greedy basado en score cuantitativo."""
    
    @staticmethod
    def generate_all_combinations() -> np.ndarray:
        """Genera todas las combinaciones posibles de 6 partidos (729)."""
        return np.array(list(itertools.product([0, 1, 2], repeat=6)))
    
    @staticmethod
    def calculate_column_metrics(combination: np.ndarray,
                                match_probabilities: List[np.ndarray],
                                match_odds: List[np.ndarray],
                                alpha: float = SystemConfig.DEFAULT_ALPHA,
                                lambda_param: float = SystemConfig.DEFAULT_LAMBDA,
                                mu_param: float = SystemConfig.DEFAULT_MU) -> Dict[str, float]:
        """
        Calcula todas las métricas para una columna S73.
        
        Returns:
            Diccionario con: entropy, adj_prob, odds, ev, dd_sens, score
        """
        # 1. Entropía de columna
        entropy = QuantitativeModel.calculate_column_entropy(
            combination, match_probabilities
        )
        
        # 2. Probabilidad conjunta ajustada
        adj_prob = QuantitativeModel.calculate_adjusted_joint_probability(
            combination, match_probabilities, entropy, alpha
        )
        
        # 3. Cuota conjunta
        odds = 1.0
        for i, sign in enumerate(combination):
            odds *= match_odds[i][sign]
        
        # 4. Expected Value
        ev = QuantitativeModel.calculate_column_ev(adj_prob, odds)
        
        # 5. Sensibilidad al drawdown
        dd_sens = QuantitativeModel.estimate_drawdown_sensitivity(odds, entropy)
        
        # 6. Score
        score = QuantitativeModel.calculate_column_score(
            ev, entropy, dd_sens, lambda_param, mu_param
        )
        
        return {
            'entropy': entropy,
            'adjusted_probability': adj_prob,
            'odds': odds,
            'expected_value': ev,
            'dd_sensitivity': dd_sens,
            'score': score,
            'combination': combination
        }
    
    @staticmethod
    def hamming_distance(comb1: np.ndarray, comb2: np.ndarray) -> int:
        """Calcula distancia de Hamming entre dos combinaciones."""
        return np.sum(comb1 != comb2)
    
    @staticmethod
    def build_optimal_s73_system(matches_data: List[MatchData],
                                target_columns: int = 35,
                                alpha: float = SystemConfig.DEFAULT_ALPHA,
                                lambda_param: float = SystemConfig.DEFAULT_LAMBDA,
                                mu_param: float = SystemConfig.DEFAULT_MU) -> List[Dict[str, Any]]:
        """
        Construye sistema S73 óptimo basado en score.
        
        Algoritmo greedy que maximiza:
        - Cobertura Hamming (≤ 2 errores)
        - Suma de scores
        - Diversificación
        
        Args:
            matches_data: Lista de objetos MatchData (debe tener 6 partidos)
            target_columns: Número objetivo de columnas (20-35 recomendado)
            alpha, lambda_param, mu_param: Hiperparámetros
            
        Returns:
            Lista de diccionarios con columnas seleccionadas y sus métricas
        """
        if len(matches_data) != 6:
            raise ValueError(f"S73 requiere exactamente 6 partidos, recibió {len(matches_data)}")
        
        match_probs = [match.prob_matrix for match in matches_data]
        match_odds = [match.odds_matrix for match in matches_data]
        
        # 1. Generar todas las combinaciones (729)
        all_combinations = OptimalS73System.generate_all_combinations()
        
        # 2. Calcular métricas para todas las combinaciones
        all_metrics = []
        for combo in all_combinations:
            metrics = OptimalS73System.calculate_column_metrics(
                combo, match_probs, match_odds, alpha, lambda_param, mu_param
            )
            all_metrics.append(metrics)
        
        # 3. Ordenar por score descendente
        all_metrics.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. Algoritmo greedy con cobertura de 2 errores
        selected_columns = []
        covered_combinations = set()
        
        # Convertir combinaciones a tuplas para usar en conjuntos
        all_combos_tuples = [tuple(combo) for combo in all_combinations]
        
        # Precalcular distancias para eficiencia
        # (optimización: solo calculamos cuando sea necesario)
        
        while len(selected_columns) < target_columns and len(all_metrics) > 0:
            best_idx = -1
            best_coverage_gain = -float('inf')
            best_score = -float('inf')
            
            # Buscar combinación que maximice cobertura de no cubiertos
            for idx, metrics in enumerate(all_metrics):
                if idx in [s['original_idx'] for s in selected_columns]:
                    continue
                
                combo_tuple = tuple(metrics['combination'])
                
                # Calcular cuántas combinaciones no cubiertas estarían cubiertas por esta
                coverage_gain = 0
                for other_tuple in all_combos_tuples:
                    if other_tuple in covered_combinations:
                        continue
                    
                    if OptimalS73System.hamming_distance(
                        np.array(combo_tuple), np.array(other_tuple)
                    ) <= SystemConfig.HAMMING_DISTANCE_TARGET:
                        coverage_gain += 1
                
                # Score combinado (cobertura + score individual)
                combined_score = coverage_gain + metrics['score'] * 100
                
                if combined_score > best_coverage_gain:
                    best_coverage_gain = combined_score
                    best_idx = idx
                    best_score = metrics['score']
            
            if best_idx == -1:
                break
            
            # Agregar columna seleccionada
            selected_metric = all_metrics[best_idx].copy()
            selected_metric['original_idx'] = best_idx
            selected_columns.append(selected_metric)
            
            # Actualizar combinaciones cubiertas
            selected_combo = selected_metric['combination']
            for other_tuple in all_combos_tuples:
                if OptimalS73System.hamming_distance(
                    selected_combo, np.array(other_tuple)
                ) <= SystemConfig.HAMMING_DISTANCE_TARGET:
                    covered_combinations.add(other_tuple)
        
        # 5. Si no alcanzamos el target, agregar las de mayor score restantes
        if len(selected_columns) < target_columns:
            remaining = [m for m in all_metrics 
                        if not any(np.array_equal(m['combination'], s['combination']) 
                                  for s in selected_columns)]
            remaining.sort(key=lambda x: x['score'], reverse=True)
            
            needed = target_columns - len(selected_columns)
            for i in range(min(needed, len(remaining))):
                selected_columns.append(remaining[i])
        
        # 6. Convertir a BetEvaluation para consistencia
        s73_evaluations = []
        for i, metrics in enumerate(selected_columns):
            bet_eval = BetEvaluation(
                bet_type=BetType.SYSTEM_S73,
                match_indices=list(range(6)),
                signs=list(metrics['combination']),
                odds=metrics['odds'],
                probability=metrics.get('raw_probability', metrics['adjusted_probability']),
                adjusted_probability=metrics['adjusted_probability'],
                entropy=metrics['entropy'],
                expected_value=metrics['expected_value'],
                dd_sensitivity=metrics['dd_sensitivity'],
                score=metrics['score']
            )
            s73_evaluations.append(bet_eval)
        
        return s73_evaluations

# ============================================================================
# SECCIÓN 5: CAPA DE ASIGNACIÓN DE CAPITAL (ALLOCATION LAYER)
# ============================================================================

class AllocationLayer:
    """Capa de asignación de capital con Kelly fraccionado ajustado."""
    
    @staticmethod
    def calculate_kelly_fraction(bet_eval: BetEvaluation,
                                bankroll: float,
                                max_kelly_fraction: float = SystemConfig.MAX_KELLY_FRACTION,
                                rho: float = SystemConfig.DEFAULT_RHO) -> float:
        """
        Calcula fracción de Kelly ajustada para una apuesta.
        
        Fórmula: f_k = min((p*q - 1)/(q - 1), f_max) * (1 - H) * ρ
        
        Args:
            bet_eval: Evaluación de la apuesta
            bankroll: Bankroll total
            max_kelly_fraction: Máximo Kelly permitido por apuesta
            rho: Factor conservador (0.3-0.7)
            
        Returns:
            Fracción del bankroll a apostar
        """
        p = bet_eval.adjusted_probability
        q = bet_eval.odds
        
        if q <= 1.0 or p <= 0:
            return 0.0
        
        # Kelly crudo
        try:
            kelly_raw = (p * q - 1) / (q - 1)
        except ZeroDivisionError:
            kelly_raw = 0.0
        
        # Aplicar límites
        kelly_capped = max(0.0, min(kelly_raw, max_kelly_fraction))
        
        # Ajustar por entropía y factor conservador
        # Para S73, usamos entropía de columna; para otros, entropía individual
        entropy_adjustment = 1.0 - bet_eval.entropy
        kelly_adjusted = kelly_capped * entropy_adjustment * rho
        
        return kelly_adjusted
    
    @staticmethod
    def normalize_portfolio_stakes(bet_evaluations: List[BetEvaluation],
                                  bankroll: float,
                                  max_exposure: float = SystemConfig.MAX_PORTFOLIO_EXPOSURE) -> List[BetEvaluation]:
        """
        Normaliza stakes para respetar exposición máxima del portafolio.
        
        Args:
            bet_evaluations: Lista de evaluaciones con kelly_fraction calculada
            bankroll: Bankroll total
            max_exposure: Exposición máxima permitida (ej: 0.2 = 20%)
            
        Returns:
            Lista de evaluaciones con stakes normalizados
        """
        # Calcular exposición total
        total_exposure = sum(eval_obj.kelly_fraction for eval_obj in bet_evaluations)
        
        if total_exposure > max_exposure:
            # Escalar proporcionalmente
            scaling_factor = max_exposure / total_exposure
            
            for eval_obj in bet_evaluations:
                eval_obj.kelly_fraction *= scaling_factor
        
        # Calcular stake en euros
        for eval_obj in bet_evaluations:
            eval_obj.stake_euros = eval_obj.kelly_fraction * bankroll
        
        return bet_evaluations

# ============================================================================
# SECCIÓN 6: CAPA DE PORTAFOLIO (PORTFOLIO LAYER)
# ============================================================================

class PortfolioLayer:
    """Capa de portafolio que integra singles, combinadas y S73."""
    
    @staticmethod
    def build_unified_portfolio(matches_data: List[MatchData],
                               bankroll: float,
                               use_singles: bool = True,
                               use_combined: bool = True,
                               use_s73: bool = True,
                               max_combined_size: int = 3,
                               s73_target_columns: int = 30,
                               alpha: float = SystemConfig.DEFAULT_ALPHA,
                               lambda_param: float = SystemConfig.DEFAULT_LAMBDA,
                               mu_param: float = SystemConfig.DEFAULT_MU,
                               rho: float = SystemConfig.DEFAULT_RHO,
                               max_exposure: float = SystemConfig.MAX_PORTFOLIO_EXPOSURE) -> Dict[str, Any]:
        """
        Construye portafolio unificado con reglas de decisión automáticas.
        
        Reglas:
        1. Singles con EV > threshold y entropía baja
        2. Combinadas solo si EV ajustado > singles equivalentes
        3. S73 como cobertura estructural y reducción de varianza
        
        Args:
            matches_data: Lista de partidos
            bankroll: Bankroll total
            use_singles: Incluir singles
            use_combined: Incluir combinadas
            use_s73: Incluir sistema S73
            max_combined_size: Tamaño máximo de combinadas
            s73_target_columns: Número objetivo de columnas S73
            alpha, lambda_param, mu_param, rho: Hiperparámetros
            max_exposure: Exposición máxima
            
        Returns:
            Diccionario con portafolio completo
        """
        portfolio = {
            'singles': [],
            'combined': [],
            's73': [],
            'total_exposure': 0.0,
            'expected_return': 0.0,
            'portfolio_score': 0.0
        }
        
        all_bets = []
        
        # 1. Evaluar todas las apuestas posibles
        if use_singles or use_combined:
            evaluator = EvaluationLayer(matches_data)
            all_evaluations = evaluator.evaluate_all_bets(
                max_combinations=max_combined_size,
                alpha=alpha,
                lambda_param=lambda_param,
                mu_param=mu_param
            )
            
            # Filtrar por tipo y calidad
            if use_singles:
                singles = [e for e in all_evaluations if e.bet_type == BetType.SINGLE]
                singles = PortfolioLayer._filter_singles(singles)
                portfolio['singles'] = singles
                all_bets.extend(singles)
            
            if use_combined:
                combined = [e for e in all_evaluations if e.bet_type in [BetType.DOUBLE, BetType.TRIPLE]]
                combined = PortfolioLayer._filter_combined(combined, singles if use_singles else [])
                portfolio['combined'] = combined
                all_bets.extend(combined)
        
        # 2. Sistema S73 (si se solicita y hay exactamente 6 partidos)
        if use_s73 and len(matches_data) >= 6:
            # Usar primeros 6 partidos para S73
            s73_matches = matches_data[:6]
            s73_evaluations = OptimalS73System.build_optimal_s73_system(
                s73_matches,
                target_columns=s73_target_columns,
                alpha=alpha,
                lambda_param=lambda_param,
                mu_param=mu_param
            )
            portfolio['s73'] = s73_evaluations
            all_bets.extend(s73_evaluations)
        
        # 3. Calcular Kelly fractions
        for bet in all_bets:
            bet.kelly_fraction = AllocationLayer.calculate_kelly_fraction(
                bet, bankroll, SystemConfig.MAX_KELLY_FRACTION, rho
            )
        
        # 4. Normalizar exposición total
        all_bets = AllocationLayer.normalize_portfolio_stakes(
            all_bets, bankroll, max_exposure
        )
        
        # 5. Calcular métricas del portafolio
        total_exposure = sum(b.kelly_fraction for b in all_bets)
        expected_return = sum(b.kelly_fraction * bankroll * b.expected_value for b in all_bets)
        portfolio_score = sum(b.score * b.kelly_fraction for b in all_bets) / total_exposure if total_exposure > 0 else 0
        
        portfolio.update({
            'all_bets': all_bets,
            'total_exposure': total_exposure,
            'expected_return': expected_return,
            'portfolio_score': portfolio_score,
            'bankroll': bankroll
        })
        
        return portfolio
    
    @staticmethod
    def _filter_singles(singles: List[BetEvaluation]) -> List[BetEvaluation]:
        """Filtra singles según criterios de calidad."""
        filtered = []
        
        for single in singles:
            # Criterios de aceptación
            if (single.expected_value >= SystemConfig.EV_THRESHOLD_SINGLE and
                single.entropy <= SystemConfig.ENTROPY_THRESHOLD and
                single.score > 0):
                filtered.append(single)
        
        # Ordenar por score descendente
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        # Limitar número de singles (ej: máximo 5)
        return filtered[:5]
    
    @staticmethod
    def _filter_combined(combined: List[BetEvaluation], 
                        singles: List[BetEvaluation]) -> List[BetEvaluation]:
        """Filtra combinadas según criterios de calidad."""
        filtered = []
        
        # Calcular EV promedio de singles para comparación
        avg_single_ev = np.mean([s.expected_value for s in singles]) if singles else 0
        
        for combo in combined:
            # Criterios más estrictos para combinadas
            if (combo.expected_value >= SystemConfig.EV_THRESHOLD_COMBINED and
                combo.entropy <= SystemConfig.ENTROPY_THRESHOLD * 0.8 and  # Más estricto
                combo.score > avg_single_ev * 1.5 and  # Mejor que singles
                combo.score > 0):
                filtered.append(combo)
        
        # Ordenar por score descendente
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        # Limitar número de combinadas (ej: máximo 3)
        return filtered[:3]

# ============================================================================
# SECCIÓN 7: INTERFAZ STREAMLIT PROFESIONAL
# ============================================================================

class ACBEQuantApp:
    """Aplicación Streamlit para ACBE v3.0."""
    
    def __init__(self):
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configuración de la página."""
        st.set_page_config(
            page_title="ACBE Quantum Betting Suite v3.0",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Renderiza sidebar con controles avanzados."""
        with st.sidebar:
            st.header("⚙️ Configuración Cuantitativa")
            
            # Bankroll y exposición
            bankroll = st.number_input(
                "Bankroll Total (€)",
                min_value=100.0,
                max_value=1000000.0,
                value=SystemConfig.DEFAULT_BANKROLL,
                step=1000.0
            )
            
            max_exposure = st.slider(
                "Exposición Máxima (%)",
                min_value=5,
                max_value=40,
                value=20,
                step=1
            ) / 100
            
            # Hiperparámetros del modelo
            st.subheader("📐 Hiperparámetros del Modelo")
            
            col1, col2 = st.columns(2)
            with col1:
                lambda_param = st.slider("λ (Entropía)", 0.0, 1.0, SystemConfig.DEFAULT_LAMBDA, 0.05)
                mu_param = st.slider("μ (Drawdown)", 0.0, 1.0, SystemConfig.DEFAULT_MU, 0.05)
            with col2:
                alpha_param = st.slider("α (Anti-sobreopt.)", 0.0, 1.0, SystemConfig.DEFAULT_ALPHA, 0.05)
                rho_param = st.slider("ρ (Conservador)", 0.0, 1.0, SystemConfig.DEFAULT_RHO, 0.05)
            
            # Componentes del portafolio
            st.subheader("🧩 Componentes del Portafolio")
            
            use_singles = st.checkbox("Incluir Singles", value=True)
            use_combined = st.checkbox("Incluir Combinadas", value=True)
            use_s73 = st.checkbox("Incluir Sistema S73", value=True)
            
            if use_combined:
                max_combined = st.slider("Máx. partidos/combinada", 2, 4, 3)
            
            if use_s73:
                s73_columns = st.slider("Columnas S73 objetivo", 20, 73, 30)
            
            # Umbrales de decisión
            st.subheader("🎯 Umbrales de Decisión")
            
            ev_single = st.slider("EV mínimo Single", 0.0, 0.3, SystemConfig.EV_THRESHOLD_SINGLE, 0.01)
            ev_combined = st.slider("EV mínimo Combinada", 0.0, 0.5, SystemConfig.EV_THRESHOLD_COMBINED, 0.01)
            entropy_max = st.slider("Entropía máxima", 0.1, 1.0, SystemConfig.ENTROPY_THRESHOLD, 0.05)
            
            # Actualizar config
            SystemConfig.EV_THRESHOLD_SINGLE = ev_single
            SystemConfig.EV_THRESHOLD_COMBINED = ev_combined
            SystemConfig.ENTROPY_THRESHOLD = entropy_max
            
            # Botón de ejecución
            st.markdown("---")
            run_analysis = st.button("🚀 Ejecutar Análisis Cuantitativo", type="primary", use_container_width=True)
            
            return {
                'bankroll': bankroll,
                'max_exposure': max_exposure,
                'lambda_param': lambda_param,
                'mu_param': mu_param,
                'alpha_param': alpha_param,
                'rho_param': rho_param,
                'use_singles': use_singles,
                'use_combined': use_combined,
                'use_s73': use_s73,
                'max_combined': max_combined if use_combined else None,
                's73_columns': s73_columns if use_s73 else None,
                'run_analysis': run_analysis
            }
    
    def render_match_input(self) -> List[MatchData]:
        """Renderiza input de partidos."""
        st.header("⚽ Input de Partidos")
        
        matches = []
        n_matches = st.number_input("Número de partidos", min_value=1, max_value=15, value=6, step=1)
        
        for i in range(n_matches):
            st.subheader(f"Partido {i+1}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                league = st.text_input(f"Liga {i+1}", value=f"Liga {i+1}", key=f"league_{i}")
                home = st.text_input(f"Local {i+1}", value=f"Equipo Local {i+1}", key=f"home_{i}")
                away = st.text_input(f"Visitante {i+1}", value=f"Equipo Visitante {i+1}", key=f"away_{i}")
            
            with col2:
                odds_1 = st.number_input(f"Cuota 1", min_value=1.01, max_value=100.0, value=2.0, key=f"odds1_{i}")
                odds_x = st.number_input(f"Cuota X", min_value=1.01, max_value=100.0, value=3.0, key=f"oddsx_{i}")
                odds_2 = st.number_input(f"Cuota 2", min_value=1.01, max_value=100.0, value=2.5, key=f"odds2_{i}")
            
            with col3:
                # Input de probabilidades (o estimación automática)
                st.markdown("**Probabilidades estimadas**")
                prob_1 = st.number_input(f"P(1)", min_value=0.0, max_value=1.0, value=0.45, step=0.01, key=f"p1_{i}")
                prob_x = st.number_input(f"P(X)", min_value=0.0, max_value=1.0, value=0.30, step=0.01, key=f"px_{i}")
                prob_2 = st.number_input(f"P(2)", min_value=0.0, max_value=1.0, value=0.25, step=0.01, key=f"p2_{i}")
            
            # Normalizar probabilidades
            total_prob = prob_1 + prob_x + prob_2
            if total_prob > 0:
                prob_1 /= total_prob
                prob_x /= total_prob
                prob_2 /= total_prob
            
            match = MatchData(
                match_id=i+1,
                league=league,
                home_team=home,
                away_team=away,
                odds_1=odds_1,
                odds_X=odds_x,
                odds_2=odds_2,
                prob_1=prob_1,
                prob_X=prob_x,
                prob_2=prob_2
            )
            matches.append(match)
            
            st.markdown("---")
        
        return matches
    
    def render_portfolio_summary(self, portfolio: Dict[str, Any]):
        """Renderiza resumen del portafolio."""
        st.header("📊 Resumen del Portafolio")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bankroll", f"€{portfolio['bankroll']:,.0f}")
            st.metric("Exposición Total", f"{portfolio['total_exposure']*100:.1f}%")
        
        with col2:
            st.metric("Retorno Esperado", f"€{portfolio['expected_return']:,.0f}")
            st.metric("ROI Esperado", f"{(portfolio['expected_return']/portfolio['bankroll'])*100:.1f}%")
        
        with col3:
            singles_count = len(portfolio['singles'])
            combined_count = len(portfolio['combined'])
            s73_count = len(portfolio['s73'])
            st.metric("Total Apuestas", singles_count + combined_count + s73_count)
            st.metric("Score Portafolio", f"{portfolio['portfolio_score']:.3f}")
        
        with col4:
            avg_ev = np.mean([b.expected_value for b in portfolio['all_bets']]) if portfolio['all_bets'] else 0
            avg_entropy = np.mean([b.entropy for b in portfolio['all_bets']]) if portfolio['all_bets'] else 0
            st.metric("EV Promedio", f"{avg_ev:.3f}")
            st.metric("Entropía Prom.", f"{avg_entropy:.3f}")
        
        # Desglose por tipo
        st.subheader("📈 Distribución por Tipo de Apuesta")
        
        if portfolio['singles']:
            st.markdown("#### 💰 Singles Recomendados")
            singles_df = pd.DataFrame([{
                'Partido': f"{b.match_indices[0]+1}: {SystemConfig.OUTCOME_LABELS[b.signs[0]]}",
                'Cuota': f"{b.odds:.2f}",
                'Prob. Ajust.': f"{b.adjusted_probability:.3f}",
                'EV': f"{b.expected_value:.3f}",
                'Entropía': f"{b.entropy:.3f}",
                'Score': f"{b.score:.3f}",
                'Kelly %': f"{b.kelly_fraction*100:.2f}%",
                'Stake (€)': f"€{b.stake_euros:.0f}"
            } for b in portfolio['singles']])
            st.dataframe(singles_df, use_container_width=True, hide_index=True)
        
        if portfolio['combined']:
            st.markdown("#### 🔗 Combinadas Recomendadas")
            combined_df = pd.DataFrame([{
                'Combinada': b.description,
                'Cuota': f"{b.odds:.2f}",
                'Prob. Ajust.': f"{b.adjusted_probability:.4f}",
                'EV': f"{b.expected_value:.3f}",
                'Entropía': f"{b.entropy:.3f}",
                'Score': f"{b.score:.3f}",
                'Kelly %': f"{b.kelly_fraction*100:.3f}%",
                'Stake (€)': f"€{b.stake_euros:.1f}"
            } for b in portfolio['combined']])
            st.dataframe(combined_df, use_container_width=True, hide_index=True)
        
        if portfolio['s73']:
            st.markdown("#### 🧮 Sistema S73 Optimizado")
            
            # Estadísticas del sistema
            s73_exposure = sum(b.kelly_fraction for b in portfolio['s73'])
            s73_expected = sum(b.kelly_fraction * portfolio['bankroll'] * b.expected_value for b in portfolio['s73'])
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Columnas S73", len(portfolio['s73']))
            with col_b:
                st.metric("Exposición S73", f"{s73_exposure*100:.1f}%")
            with col_c:
                st.metric("Retorno Esperado S73", f"€{s73_expected:.0f}")
            
            # Mostrar primeras 10 columnas
            s73_df = pd.DataFrame([{
                'Columna': ''.join([SystemConfig.OUTCOME_LABELS[s] for s in b.signs]),
                'Cuota': f"{b.odds:.2f}",
                'Prob. Ajust.': f"{b.adjusted_probability:.5f}",
                'EV': f"{b.expected_value:.3f}",
                'Entropía': f"{b.entropy:.3f}",
                'Score': f"{b.score:.3f}",
                'Kelly %': f"{b.kelly_fraction*100:.3f}%",
                'Stake (€)': f"€{b.stake_euros:.1f}"
            } for b in portfolio['s73'][:10]])
            st.dataframe(s73_df, use_container_width=True, hide_index=True)
            
            if len(portfolio['s73']) > 10:
                st.info(f"Mostrando 10 de {len(portfolio['s73'])} columnas S73. Descarga completa disponible.")
    
    def render_risk_analysis(self, portfolio: Dict[str, Any]):
        """Renderiza análisis de riesgo avanzado."""
        st.header("⚠️ Análisis de Riesgo Avanzado")
        
        if not portfolio['all_bets']:
            st.warning("No hay apuestas para analizar.")
            return
        
        bets = portfolio['all_bets']
        
        # Calcular métricas de riesgo
        exposures = [b.kelly_fraction for b in bets]
        evs = [b.expected_value for b in bets]
        entropies = [b.entropy for b in bets]
        scores = [b.score for b in bets]
        
        # Diversificación
        unique_matches = set()
        for bet in bets:
            if bet.bet_type == BetType.SYSTEM_S73:
                unique_matches.update(range(6))
            else:
                unique_matches.update(bet.match_indices)
        
        # Concentración
        herfindahl = sum(e**2 for e in exposures)
        
        # Visualizaciones
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de exposición
            fig_exposure = go.Figure(data=[go.Pie(
                labels=[f"Bet {i+1}" for i in range(len(exposures))],
                values=exposures,
                hole=0.3,
                marker_colors=SystemConfig.COLORS['info']
            )])
            fig_exposure.update_layout(title="Distribución de Exposición")
            st.plotly_chart(fig_exposure, use_container_width=True)
        
        with col2:
            # Scatter EV vs Entropía
            fig_scatter = go.Figure()
            
            # Color por tipo de apuesta
            colors = {
                BetType.SINGLE: SystemConfig.COLORS['success'],
                BetType.DOUBLE: SystemConfig.COLORS['warning'],
                BetType.TRIPLE: SystemConfig.COLORS['primary'],
                BetType.SYSTEM_S73: SystemConfig.COLORS['danger']
            }
            
            for bet_type in [BetType.SINGLE, BetType.DOUBLE, BetType.TRIPLE, BetType.SYSTEM_S73]:
                type_bets = [b for b in bets if b.bet_type == bet_type]
                if type_bets:
                    fig_scatter.add_trace(go.Scatter(
                        x=[b.entropy for b in type_bets],
                        y=[b.expected_value for b in type_bets],
                        mode='markers',
                        name=bet_type.value,
                        marker=dict(
                            size=[b.kelly_fraction * 100 for b in type_bets],
                            color=colors[bet_type],
                            opacity=0.7
                        ),
                        text=[b.description for b in type_bets]
                    ))
            
            fig_scatter.update_layout(
                title="EV vs Entropía (tamaño = exposición)",
                xaxis_title="Entropía",
                yaxis_title="Expected Value",
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Métricas de riesgo
        st.subheader("📊 Métricas de Riesgo")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Diversificación", f"{len(unique_matches)} partidos únicos")
        with col2:
            st.metric("Índice Herfindahl", f"{herfindahl:.4f}")
        with col3:
            max_exposure = max(exposures) if exposures else 0
            st.metric("Máxima Exposición", f"{max_exposure*100:.2f}%")
        with col4:
            var_95 = np.percentile(scores, 5) if scores else 0
            st.metric("VaR 95% (Score)", f"{var_95:.3f}")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones de Gestión de Riesgo")
        
        recommendations = []
        
        if herfindahl > 0.15:
            recommendations.append("⚠️ **Concentración alta**: Considera diversificar más las apuestas.")
        
        if max_exposure > 0.03:
            recommendations.append("⚠️ **Exposición individual alta**: Reduce el stake en apuestas individuales.")
        
        if len([e for e in evs if e < 0]) > len(evs) * 0.3:
            recommendations.append("⚠️ **Muchas apuestas con EV negativo**: Revisa los criterios de selección.")
        
        if not recommendations:
            recommendations.append("✅ **Perfil de riesgo equilibrado**: El portafolio está bien diversificado.")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    def run(self):
        """Método principal de ejecución."""
        st.title("🎯 ACBE Quantum Betting Suite v3.0")
        st.markdown("""
        *Sistema cuantitativo institucional para optimización de portafolios de apuestas*  
        *Arquitectura unificada: Singles + Combinadas + Sistema S73 Optimizado*
        """)
        
        # Sidebar con configuración
        config = self.render_sidebar()
        
        if not config['run_analysis']:
            st.info("👈 Configura los parámetros en la sidebar y ejecuta el análisis")
            return
        
        try:
            # Input de partidos
            matches_data = self.render_match_input()
            
            if not matches_data:
                st.warning("Por favor, ingresa al menos un partido.")
                return
            
            # Construir portafolio unificado
            with st.spinner("🔄 Construyendo portafolio cuantitativo..."):
                portfolio = PortfolioLayer.build_unified_portfolio(
                    matches_data=matches_data,
                    bankroll=config['bankroll'],
                    use_singles=config['use_singles'],
                    use_combined=config['use_combined'],
                    use_s73=config['use_s73'],
                    max_combined_size=config.get('max_combined', 3),
                    s73_target_columns=config.get('s73_columns', 30),
                    alpha=config['alpha_param'],
                    lambda_param=config['lambda_param'],
                    mu_param=config['mu_param'],
                    rho=config['rho_param'],
                    max_exposure=config['max_exposure']
                )
            
            # Mostrar resultados en pestañas
            tab1, tab2, tab3 = st.tabs(["📊 Portafolio", "⚠️ Riesgo", "📈 Análisis Detallado"])
            
            with tab1:
                self.render_portfolio_summary(portfolio)
            
            with tab2:
                self.render_risk_analysis(portfolio)
            
            with tab3:
                self.render_detailed_analysis(matches_data, portfolio)
                
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.exception(e)
    
    def render_detailed_analysis(self, matches_data: List[MatchData], portfolio: Dict[str, Any]):
        """Renderiza análisis detallado."""
        st.header("📈 Análisis Detallado")
        
        # Análisis de partidos individuales
        st.subheader("⚽ Análisis por Partido")
        
        matches_df = pd.DataFrame([{
            'Partido': i+1,
            'Local': m.home_team,
            'Visitante': m.away_team,
            'P(1)': f"{m.prob_1:.3f}",
            'P(X)': f"{m.prob_X:.3f}",
            'P(2)': f"{m.prob_2:.3f}",
            'Cuota 1': f"{m.odds_1:.2f}",
            'Cuota X': f"{m.odds_X:.2f}",
            'Cuota 2': f"{m.odds_2:.2f}",
            'EV 1': f"{m.ev_matrix[0]:.3f}",
            'EV X': f"{m.ev_matrix[1]:.3f}",
            'EV 2': f"{m.ev_matrix[2]:.3f}",
            'Entropía': f"{m.entropy:.3f}"
        } for i, m in enumerate(matches_data)])
        
        st.dataframe(matches_df, use_container_width=True, hide_index=True)
        
        # Gráfico de probabilidades
        fig_probs = go.Figure()
        for i, match in enumerate(matches_data):
            fig_probs.add_trace(go.Bar(
                x=[f"Partido {i+1}"],
                y=[match.prob_1],
                name='1',
                marker_color=SystemConfig.OUTCOME_COLORS[0],
                showlegend=(i==0)
            ))
            fig_probs.add_trace(go.Bar(
                x=[f"Partido {i+1}"],
                y=[match.prob_X],
                name='X',
                marker_color=SystemConfig.OUTCOME_COLORS[1],
                showlegend=(i==0)
            ))
            fig_probs.add_trace(go.Bar(
                x=[f"Partido {i+1}"],
                y=[match.prob_2],
                name='2',
                marker_color=SystemConfig.OUTCOME_COLORS[2],
                showlegend=(i==0)
            ))
        
        fig_probs.update_layout(
            title="Probabilidades por Partido",
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig_probs, use_container_width=True)
        
        # Descarga de resultados
        st.subheader("💾 Exportar Resultados")
        
        if st.button("📥 Descargar Portafolio Completo (CSV)"):
            # Crear DataFrame para descarga
            download_data = []
            
            for bet in portfolio['all_bets']:
                download_data.append({
                    'Tipo': bet.bet_type.value,
                    'Descripción': bet.description,
                    'Cuota': bet.odds,
                    'Probabilidad_Cruda': bet.probability,
                    'Probabilidad_Ajustada': bet.adjusted_probability,
                    'Expected_Value': bet.expected_value,
                    'Entropía': bet.entropy,
                    'Score': bet.score,
                    'Kelly_Fraction': bet.kelly_fraction,
                    'Stake_Euros': bet.stake_euros,
                    'Retorno_Esperado': bet.stake_euros * bet.expected_value
                })
            
            df_download = pd.DataFrame(download_data)
            csv = df_download.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV",
                data=csv,
                file_name="acbe_portfolio.csv",
                mime="text/csv"
            )

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    app = ACBEQuantApp()
    app.run()
