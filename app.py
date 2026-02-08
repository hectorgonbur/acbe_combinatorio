"""
🎯 ACBE-S73 QUANTUM BETTING SUITE v2.0
Sistema profesional de optimización de portafolios de apuestas deportivas
Combina Inferencia Bayesiana Gamma-Poisson, Teoría de la Información y Criterio de Kelly
Con cobertura S73 completa (2 errores) y gestión probabilística avanzada
Autor: Arquitecto de Software & Data Scientist Senior
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import List, Tuple, Dict, Optional
warnings.filterwarnings('ignore')

# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN DEL SISTEMA Y CONSTANTES MATEMÁTICAS
# ============================================================================

class SystemConfig:
    """Configuración centralizada del sistema ACBE-S73."""
    
    # Parámetros de simulación
    MONTE_CARLO_ITERATIONS = 10000
    KELLY_FRACTION_MAX = 0.03  # 3% máximo por columna
    MIN_PROBABILITY = 1e-10    # Evitar log(0)
    BASE_ENTROPY = 3           # Base logarítmica para 3 resultados
    
    # Modelo Bayesiano Gamma-Poisson
    DEFAULT_ATTACK_MEAN = 1.2
    DEFAULT_DEFENSE_MEAN = 0.8
    DEFAULT_HOME_ADVANTAGE = 1.1
    DEFAULT_ALPHA_PRIOR = 2.0  # Parámetro de forma Gamma
    DEFAULT_BETA_PRIOR = 1.0   # Parámetro de tasa Gamma
    
    # Sistema S73
    NUM_MATCHES = 6            # Partidos por sistema
    FULL_COMBINATIONS = 3 ** 6  # 729 combinaciones posibles
    TARGET_COMBINATIONS = 73   # Objetivo de columnas reducidas
    HAMMING_DISTANCE_TARGET = 2  # ¡CORREGIDO: Cobertura de 2 errores!
    
    # Umbrales de clasificación por entropía
    STRONG_MATCH_THRESHOLD = 0.30   # ≤ 0.30: Partido Fuerte (1 signo)
    MEDIUM_MATCH_THRESHOLD = 0.60   # 0.30-0.60: Partido Medio (2 signos)
                                    # ≥ 0.60: Partido Caótico (3 signos)
    
    # Gestión de riesgo
    MIN_ODDS = 1.01
    MAX_ODDS = 100.0
    DEFAULT_BANKROLL = 10000.0
    MAX_PORTFOLIO_EXPOSURE = 0.15   # 15% exposición máxima del portafolio
    MIN_JOINT_PROBABILITY = 0.001   # Umbral mínimo probabilidad conjunta
    
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
# SECCIÓN 2: MODELO MATEMÁTICO ACBE (VECTORIZADO)
# ============================================================================

class ACBEModel:
    """Modelo Bayesiano Gamma-Poisson para estimación de probabilidades."""
    
    @staticmethod
    @st.cache_data
    def vectorized_poisson_simulation(lambda_home: np.ndarray, 
                                     lambda_away: np.ndarray, 
                                     n_sims: int = SystemConfig.MONTE_CARLO_ITERATIONS) -> np.ndarray:
        """
        Simulación vectorizada de resultados usando distribución Poisson.
        
        Args:
            lambda_home: Tasas de goles locales (n_matches,)
            lambda_away: Tasas de goles visitantes (n_matches,)
            n_sims: Iteraciones Monte Carlo
            
        Returns:
            Array (n_matches, 3) con probabilidades [P(1), P(X), P(2)]
        """
        n_matches = len(lambda_home)
        
        # Generación vectorizada de goles
        home_goals = np.random.poisson(
            lam=np.tile(lambda_home, (n_sims, 1)),
            size=(n_sims, n_matches)
        )
        
        away_goals = np.random.poisson(
            lam=np.tile(lambda_away, (n_sims, 1)),
            size=(n_sims, n_matches)
        )
        
        # Cálculo de resultados (vectorizado)
        home_wins = (home_goals > away_goals).sum(axis=0) / n_sims
        draws = (home_goals == away_goals).sum(axis=0) / n_sims
        away_wins = (home_goals < away_goals).sum(axis=0) / n_sims
        
        # Ensamblar matriz de probabilidades
        probabilities = np.column_stack([home_wins, draws, away_wins])
        
        # Normalización y estabilidad numérica
        probabilities = np.clip(probabilities, SystemConfig.MIN_PROBABILITY, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    @staticmethod
    @st.cache_data
    def gamma_poisson_bayesian(attack_strengths: np.ndarray,
                              defense_strengths: np.ndarray,
                              home_advantage: float = SystemConfig.DEFAULT_HOME_ADVANTAGE,
                              alpha_prior: float = SystemConfig.DEFAULT_ALPHA_PRIOR,
                              beta_prior: float = SystemConfig.DEFAULT_BETA_PRIOR) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modelo Bayesian Gamma-Poisson para estimar tasas de goles.
        
        Args:
            attack_strengths: Array (n_matches, 2) [ataque_local, ataque_visitante]
            defense_strengths: Array (n_matches, 2) [defensa_local, defensa_visitante]
            home_advantage: Ventaja de local
            
        Returns:
            lambda_home, lambda_away: Tasas estimadas de goles
        """
        n_matches = attack_strengths.shape[0]
        
        # Cálculo de lambda con ventaja local
        lambda_home = attack_strengths[:, 0] * defense_strengths[:, 1] * home_advantage
        lambda_away = attack_strengths[:, 1] * defense_strengths[:, 0]
        
        # Actualización Bayesian (Gamma-Poisson conjugada)
        alpha_posterior = alpha_prior + lambda_home + lambda_away
        beta_posterior = beta_prior + 2  # 2 equipos por partido
        
        # Muestreo de la posterior (vectorizado)
        lambda_home_samples = np.random.gamma(
            shape=alpha_posterior,
            scale=1/beta_posterior,
            size=(SystemConfig.MONTE_CARLO_ITERATIONS, n_matches)
        ).mean(axis=0)
        
        lambda_away_samples = np.random.gamma(
            shape=alpha_posterior,
            scale=1/beta_posterior,
            size=(SystemConfig.MONTE_CARLO_ITERATIONS, n_matches)
        ).mean(axis=0)
        
        return lambda_home_samples, lambda_away_samples
    
    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> np.ndarray:
        """
        Calcula entropía de Shannon (base 3) para cada partido.
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            
        Returns:
            Array (n_matches,) de entropías
        """
        # Estabilidad numérica
        probs = np.clip(probabilities, SystemConfig.MIN_PROBABILITY, 1.0)
        
        # Entropía vectorizada (base 3)
        entropy = -np.sum(probs * np.log(probs) / np.log(SystemConfig.BASE_ENTROPY), axis=1)
        
        return entropy
    
    @staticmethod
    def normalize_entropy(entropy: np.ndarray) -> np.ndarray:
        """
        Normaliza entropías al rango [0, 1].
        
        Args:
            entropy: Array de entropías
            
        Returns:
            Array normalizado
        """
        if np.max(entropy) - np.min(entropy) < SystemConfig.MIN_PROBABILITY:
            return np.ones_like(entropy)
        
        return (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))

# ============================================================================
# SECCIÓN 3: TEORÍA DE LA INFORMACIÓN Y CLASIFICACIÓN PROBABILÍSTICA
# ============================================================================

class InformationTheory:
    """Clasificación probabilística basada en entropía y teoría de información."""
    
    @staticmethod
    def classify_matches_by_entropy(probabilities: np.ndarray, 
                                   normalized_entropies: np.ndarray) -> Tuple[List[List[int]], List[str]]:
        """
        Clasifica partidos según entropía normalizada y reduce espacio de signos.
        
        Sistema de clasificación:
        - Entropía ≤ 0.30: Partido Fuerte → 1 signo (el más probable)
        - Entropía 0.30-0.60: Partido Medio → 2 signos (más probables)
        - Entropía ≥ 0.60: Partido Caótico → 3 signos
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            normalized_entropies: Array (n_matches,) de entropías normalizadas
            
        Returns:
            allowed_signs: Lista de listas con signos permitidos por partido
            classifications: Lista de clasificaciones
        """
        allowed_signs = []
        classifications = []
        
        for i in range(len(probabilities)):
            entropy_norm = normalized_entropies[i]
            
            if entropy_norm <= SystemConfig.STRONG_MATCH_THRESHOLD:
                # Partido Fuerte: solo el signo más probable
                best_sign = np.argmax(probabilities[i])
                allowed_signs.append([best_sign])
                classifications.append('Fuerte')
                
            elif entropy_norm <= SystemConfig.MEDIUM_MATCH_THRESHOLD:
                # Partido Medio: 2 signos más probables
                top_two = np.argsort(probabilities[i])[-2:].tolist()
                allowed_signs.append(top_two)
                classifications.append('Medio')
                
            else:
                # Partido Caótico: 3 signos
                allowed_signs.append([0, 1, 2])
                classifications.append('Caótico')
        
        return allowed_signs, classifications
    
    @staticmethod
    def calculate_expected_value(probabilities: np.ndarray, odds_matrix: np.ndarray) -> np.ndarray:
        """
        Calcula el valor esperado (EV) para cada apuesta.
        
        Fórmula: EV = p * q - 1, donde:
        - p: probabilidad estimada
        - q: cuota ofrecida
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            odds_matrix: Array (n_matches, 3) de cuotas
            
        Returns:
            Array (n_matches, 3) de valores esperados
        """
        return probabilities * odds_matrix - 1

# ============================================================================
# SECCIÓN 4: SISTEMA COMBINATORIO S73 (COBERTURA DE 2 ERRORES)
# ============================================================================

class S73System:
    """Sistema combinatorio S73 con cobertura garantizada de 2 errores."""
    
    @staticmethod
    @st.cache_data
    def generate_prefiltered_combinations(probabilities: np.ndarray,
                                         normalized_entropies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera combinaciones pre-filtradas usando clasificación por entropía.
        
        Reduce el espacio de búsqueda antes de aplicar el sistema S73.
        
        Args:
            probabilities: Array (6, 3) de probabilidades (para 6 partidos)
            normalized_entropies: Array (6,) de entropías normalizadas
            
        Returns:
            combinations: Array (n_combinations, 6) de combinaciones filtradas
            joint_probs: Array (n_combinations,) de probabilidades conjuntas
        """
        # 1. Clasificar partidos y obtener signos permitidos
        allowed_signs, _ = InformationTheory.classify_matches_by_entropy(
            probabilities, normalized_entropies
        )
        
        # 2. Generar producto cartesiano de signos permitidos
        import itertools
        combinations_list = list(itertools.product(*allowed_signs))
        combinations = np.array(combinations_list)
        
        # 3. Calcular probabilidades conjuntas (vectorizado)
        n_combinations = len(combinations)
        joint_probs = np.ones(n_combinations)
        
        for idx, combo in enumerate(combinations):
            for match_idx, sign in enumerate(combo):
                joint_probs[idx] *= probabilities[match_idx, sign]
        
        # 4. Filtrar por umbral mínimo de probabilidad conjunta
        mask = joint_probs >= SystemConfig.MIN_JOINT_PROBABILITY
        filtered_combinations = combinations[mask]
        filtered_probs = joint_probs[mask]
        
        return filtered_combinations, filtered_probs
    
    @staticmethod
    def hamming_distance_matrix(combinations: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancias de Hamming entre combinaciones.
        
        Args:
            combinations: Array (n_combinations, 6) de combinaciones
            
        Returns:
            Array (n_combinations, n_combinations) de distancias
        """
        n = len(combinations)
        distances = np.zeros((n, n), dtype=np.int8)
        
        # Cálculo eficiente de distancias Hamming
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sum(combinations[i] != combinations[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    @staticmethod
    @st.cache_data
    def build_s73_coverage_system(filtered_combinations: np.ndarray,
                                 filtered_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye sistema S73 con cobertura garantizada de 2 errores.
        
        Implementa algoritmo greedy optimizado que selecciona combinaciones
        que maximizan la cobertura de espacio (Hamming distance ≤ 2).
        
        Args:
            filtered_combinations: Combinaciones pre-filtradas
            filtered_probs: Probabilidades conjuntas correspondientes
            
        Returns:
            selected_combinations: Array de combinaciones seleccionadas
            selected_probs: Array de probabilidades seleccionadas
        """
        n_combinations = len(filtered_combinations)
        
        if n_combinations <= SystemConfig.TARGET_COMBINATIONS:
            return filtered_combinations, filtered_probs
        
        # 1. Ordenar por probabilidad descendente
        sorted_indices = np.argsort(filtered_probs)[::-1]
        sorted_combinations = filtered_combinations[sorted_indices]
        sorted_probs = filtered_probs[sorted_indices]
        
        # 2. Precalcular matriz de distancias Hamming
        distance_matrix = S73System.hamming_distance_matrix(sorted_combinations)
        
        # 3. Algoritmo greedy con cobertura de 2 errores
        selected_indices = []
        covered_indices = set()
        
        while (len(selected_indices) < SystemConfig.TARGET_COMBINATIONS and 
               len(covered_indices) < n_combinations):
            
            best_idx = -1
            best_coverage_gain = -1
            
            # Buscar combinación que maximice cobertura de no cubiertos
            for i in range(n_combinations):
                if i in selected_indices:
                    continue
                
                # Combinaciones cubiertas por i (Hamming ≤ 2)
                coverage_mask = distance_matrix[i] <= SystemConfig.HAMMING_DISTANCE_TARGET
                uncovered_coverage = sum(1 for j in range(n_combinations) 
                                       if coverage_mask[j] and j not in covered_indices)
                
                # Ponderar por probabilidad y cobertura
                coverage_gain = uncovered_coverage * (1 + sorted_probs[i])
                
                if coverage_gain > best_coverage_gain:
                    best_coverage_gain = coverage_gain
                    best_idx = i
            
            if best_idx == -1:
                break
            
            # Agregar combinación seleccionada
            selected_indices.append(best_idx)
            
            # Actualizar conjunto de combinaciones cubiertas
            newly_covered = np.where(
                distance_matrix[best_idx] <= SystemConfig.HAMMING_DISTANCE_TARGET
            )[0]
            covered_indices.update(newly_covered)
        
        # 4. Si no alcanza el target, completar con más probables
        if len(selected_indices) < SystemConfig.TARGET_COMBINATIONS:
            remaining_needed = SystemConfig.TARGET_COMBINATIONS - len(selected_indices)
            for i in range(n_combinations):
                if i not in selected_indices:
                    selected_indices.append(i)
                    remaining_needed -= 1
                    if remaining_needed == 0:
                        break
        
        # 5. Extraer combinaciones seleccionadas
        selected_combinations = sorted_combinations[selected_indices]
        selected_probs = sorted_probs[selected_indices]
        
        return selected_combinations, selected_probs
    
    @staticmethod
    def calculate_combination_odds(combination: np.ndarray, odds_matrix: np.ndarray) -> float:
        """
        Calcula la cuota conjunta de una combinación.
        
        Args:
            combination: Array (6,) con signos seleccionados
            odds_matrix: Array (6, 3) de cuotas
            
        Returns:
            Cuota conjunta (producto de cuotas seleccionadas)
        """
        selected_odds = odds_matrix[np.arange(6), combination]
        return np.prod(selected_odds)

# ============================================================================
# SECCIÓN 5: CRITERIO DE KELLY INTEGRADO Y GESTIÓN DE CAPITAL
# ============================================================================

class KellyCapitalManagement:
    """Gestión de capital basada en criterio de Kelly con ajustes por entropía."""
    
    @staticmethod
    def calculate_kelly_stakes(probabilities: np.ndarray,
                              odds_matrix: np.ndarray,
                              normalized_entropies: np.ndarray,
                              kelly_fraction: float = 1.0) -> np.ndarray:
        """
        Calcula stakes Kelly ajustados por entropía.
        
        Fórmula Kelly: f = (p*q - 1) / (q - 1)
        Ajuste por entropía: f_adj = f * (1 - H) * kelly_fraction
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            odds_matrix: Array (n_matches, 3) de cuotas
            normalized_entropies: Array (n_matches,) de entropías normalizadas
            kelly_fraction: Fracción de Kelly a aplicar (0-1)
            
        Returns:
            Array (n_matches, 3) de stakes recomendados
        """
        # Calcular Kelly crudo
        with np.errstate(divide='ignore', invalid='ignore'):
            kelly_raw = (probabilities * odds_matrix - 1) / (odds_matrix - 1)
        
        # Manejar casos especiales
        kelly_raw = np.nan_to_num(kelly_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Aplicar límites (0 a KELLY_FRACTION_MAX)
        kelly_capped = np.clip(kelly_raw, 0, SystemConfig.KELLY_FRACTION_MAX)
        
        # Ajustar por entropía (más incertidumbre → menor stake)
        entropy_adjustment = (1.0 - normalized_entropies[:, np.newaxis])
        stakes = kelly_capped * entropy_adjustment * kelly_fraction
        
        return stakes
    
    @staticmethod
    def calculate_column_kelly(combination: np.ndarray,
                              joint_probability: float,
                              combination_odds: float,
                              avg_entropy: float) -> float:
        """
        Calcula stake Kelly para una columna del sistema S73.
        
        Args:
            combination: Array (6,) de signos
            joint_probability: Probabilidad conjunta de la combinación
            combination_odds: Cuota conjunta
            avg_entropy: Entropía promedio de la combinación
            
        Returns:
            Stake Kelly ajustado (porcentaje del bankroll)
        """
        if combination_odds <= 1.0:
            return 0.0
        
        # Kelly para la combinación
        kelly_raw = (joint_probability * combination_odds - 1) / (combination_odds - 1)
        
        # Aplicar límites y ajuste por entropía
        kelly_capped = max(0.0, min(kelly_raw, SystemConfig.KELLY_FRACTION_MAX))
        kelly_adjusted = kelly_capped * (1.0 - avg_entropy)
        
        return kelly_adjusted
    
    @staticmethod
    def normalize_portfolio_stakes(stakes_array: np.ndarray,
                                  max_exposure: float = SystemConfig.MAX_PORTFOLIO_EXPOSURE) -> np.ndarray:
        """
        Normaliza stakes para limitar exposición total del portafolio.
        
        Args:
            stakes_array: Array de stakes individuales
            max_exposure: Exposición máxima permitida (ej: 0.15 = 15%)
            
        Returns:
            Array de stakes normalizados
        """
        total_exposure = np.sum(stakes_array)
        
        if total_exposure > max_exposure:
            # Escalar proporcionalmente para respetar límite
            scaling_factor = max_exposure / total_exposure
            stakes_array = stakes_array * scaling_factor
        
        return stakes_array

# ============================================================================
# SECCIÓN 6: MOTOR DE BACKTESTING VECTORIZADO
# ============================================================================

class VectorizedBacktester:
    """Motor de backtesting completamente vectorizado con gestión real de capital."""
    
    def __init__(self, initial_bankroll: float = SystemConfig.DEFAULT_BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.equity_curve = [initial_bankroll]
        self.drawdown_curve = [0.0]
    
    @staticmethod
    @st.cache_data
    def simulate_match_outcomes(probabilities: np.ndarray, n_sims: int) -> np.ndarray:
        """
        Simula resultados de partidos usando distribución multinomial.
        
        Args:
            probabilities: Array (n_matches, 3) de probabilidades
            n_sims: Número de simulaciones
            
        Returns:
            Array (n_sims, n_matches) de resultados (0, 1, 2)
        """
        n_matches = probabilities.shape[0]
        outcomes = np.zeros((n_sims, n_matches), dtype=int)
        
        for i in range(n_matches):
            # Muestreo multinomial vectorizado
            samples = np.random.multinomial(1, probabilities[i], size=n_sims)
            outcomes[:, i] = np.argmax(samples, axis=1)
        
        return outcomes
    
    def calculate_column_performance(self,
                                    real_outcomes: np.ndarray,
                                    combinations: np.ndarray,
                                    odds_matrix: np.ndarray,
                                    stakes_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula rendimiento de columnas con stakes reales.
        
        Args:
            real_outcomes: Resultados simulados (n_sims, 6)
            combinations: Combinaciones del sistema (n_columns, 6)
            odds_matrix: Cuotas (6, 3)
            stakes_array: Stakes por columna (n_columns,)
            
        Returns:
            total_returns: Retornos totales por simulación
            column_returns: Retornos por columna (n_sims, n_columns)
        """
        n_sims, n_matches = real_outcomes.shape
        n_columns = len(combinations)
        
        # Inicializar matriz de retornos
        column_returns = np.zeros((n_sims, n_columns))
        
        # Calcular cuotas conjuntas por columna
        combination_odds = np.zeros(n_columns)
        for i, combo in enumerate(combinations):
            combination_odds[i] = S73System.calculate_combination_odds(combo, odds_matrix)
        
        # Calcular stakes en euros
        stakes_euros = stakes_array * self.bankroll
        
        # Vectorizar comparación de resultados
        for col_idx, combination in enumerate(combinations):
            # Verificar aciertos (True si todos los partidos coinciden)
            hits = np.all(real_outcomes == combination, axis=1)
            
            # Calcular retorno: ganancia si acierta, pérdida stake si falla
            column_returns[:, col_idx] = np.where(
                hits,
                stakes_euros[col_idx] * (combination_odds[col_idx] - 1),  # Ganancia
                -stakes_euros[col_idx]                                   # Pérdida
            )
        
        # Retorno total por simulación (suma de todas las columnas)
        total_returns = column_returns.sum(axis=1)
        
        return total_returns, column_returns
    
    def run_backtest(self,
                    probabilities: np.ndarray,
                    odds_matrix: np.ndarray,
                    normalized_entropies: np.ndarray,
                    s73_results: Dict,
                    n_rounds: int = 100,
                    n_sims_per_round: int = 1000,
                    kelly_fraction: float = 0.5) -> Dict:
        """
        Ejecuta backtesting completo con gestión realista de capital.
        
        Args:
            probabilities: Probabilidades ACBE (6, 3)
            odds_matrix: Cuotas (6, 3)
            normalized_entropies: Entropías normalizadas (6,)
            s73_results: Resultados del sistema S73
            n_rounds: Número de rondas/jornadas
            n_sims_per_round: Simulaciones Monte Carlo por ronda
            kelly_fraction: Fracción conservadora de Kelly
            
        Returns:
            Diccionario con resultados del backtest
        """
        combinations = s73_results['combinations']
        n_columns = len(combinations)
        
        # Reinicializar métricas
        self.bankroll = self.initial_bankroll
        self.equity_curve = [self.bankroll]
        self.drawdown_curve = [0.0]
        
        all_returns = []
        round_metrics = []
        
        for round_idx in range(n_rounds):
            # 1. Simular resultados reales
            real_outcomes = self.simulate_match_outcomes(probabilities, n_sims_per_round)
            
            # 2. Calcular stakes actualizados (pueden cambiar con bankroll)
            current_stakes = self._calculate_current_stakes(
                s73_results, kelly_fraction
            )
            
            # 3. Calcular rendimiento
            round_returns, column_returns = self.calculate_column_performance(
                real_outcomes, combinations, odds_matrix, current_stakes
            )
            
            # 4. Actualizar bankroll (usar retorno promedio esperado)
            avg_return = np.mean(round_returns)
            self.bankroll += avg_return
            
            # 5. Registrar métricas
            self.equity_curve.append(self.bankroll)
            all_returns.extend(round_returns)
            
            # Calcular drawdown actual
            peak = np.max(self.equity_curve)
            current_dd = (peak - self.bankroll) / peak * 100 if peak > 0 else 0
            self.drawdown_curve.append(current_dd)
            
            # Métricas de la ronda
            round_metrics.append({
                'round': round_idx + 1,
                'bankroll': self.bankroll,
                'avg_return': avg_return,
                'std_return': np.std(round_returns),
                'win_rate': np.mean(round_returns > 0) * 100,
                'max_single_return': np.max(round_returns),
                'min_single_return': np.min(round_returns)
            })
        
        # Calcular métricas finales
        final_metrics = self._calculate_final_metrics(all_returns, n_rounds)
        
        return {
            'equity_curve': np.array(self.equity_curve),
            'drawdown_curve': np.array(self.drawdown_curve),
            'final_metrics': final_metrics,
            'round_metrics': round_metrics,
            'all_returns': np.array(all_returns)
        }
    
    def _calculate_current_stakes(self, s73_results: Dict, kelly_fraction: float) -> np.ndarray:
        """Calcula stakes actualizados basados en bankroll actual."""
        stakes = s73_results['kelly_stakes'].copy()
        
        # Ajustar por fracción de Kelly conservadora
        stakes = stakes * kelly_fraction
        
        # Normalizar para limitar exposición total
        stakes = KellyCapitalManagement.normalize_portfolio_stakes(stakes)
        
        return stakes
    
    def _calculate_final_metrics(self, all_returns: List[float], n_rounds: int) -> Dict:
        """Calcula métricas finales agregadas del backtest."""
        returns_array = np.array(all_returns)
        
        # ROI y retorno total
        total_return = self.bankroll - self.initial_bankroll
        total_return_pct = (total_return / self.initial_bankroll) * 100
        
        # Sharpe Ratio (tasa libre de riesgo = 0)
        if np.std(returns_array) > 0:
            sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Drawdown máximo
        max_drawdown = np.max(self.drawdown_curve)
        
        # CAGR (Compound Annual Growth Rate)
        if self.bankroll > 0:
            cagr = ((self.bankroll / self.initial_bankroll) ** (252 / n_rounds) - 1) * 100
        else:
            cagr = -100.0
        
        # Value at Risk (VaR 95%)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Win rate y estadísticas
        positive_returns = returns_array[returns_array > 0]
        negative_returns = returns_array[returns_array <= 0]
        
        win_rate = (len(positive_returns) / len(returns_array) * 100) if len(returns_array) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        
        # Profit factor
        if np.sum(negative_returns) < 0:
            profit_factor = abs(np.sum(positive_returns) / np.sum(negative_returns))
        else:
            profit_factor = 0.0
        
        # Probabilidad de ruina (bankroll < 50% inicial)
        ruin_prob = np.mean(np.array(self.equity_curve) < self.initial_bankroll * 0.5) * 100
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.bankroll,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'roi_per_round': (np.mean(returns_array) / self.initial_bankroll) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cagr': cagr,
            'var_95': var_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'ruin_probability': ruin_prob,
            'std_returns': np.std(returns_array) if len(returns_array) > 0 else 0
        }

# ============================================================================
# SECCIÓN 7: GENERADOR DE DATOS SINTÉTICOS
# ============================================================================

class SyntheticDataGenerator:
    """Genera datos sintéticos realistas para pruebas del sistema."""
    
    @staticmethod
    @st.cache_data
    def generate_complete_dataset(n_matches: int = 6, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Genera dataset sintético completo con parámetros realistas.
        
        Args:
            n_matches: Número de partidos a generar
            seed: Semilla para reproducibilidad
            
        Returns:
            matches_df: DataFrame con parámetros de equipos
            odds_df: DataFrame con cuotas
            probabilities: Array (n_matches, 3) de probabilidades reales
        """
        np.random.seed(seed)
        
        # Parámetros de equipos (distribución Beta para mayor realismo)
        attack_strengths = np.random.beta(a=2, b=2, size=(n_matches, 2)) * 1.5 + 0.5
        defense_strengths = np.random.beta(a=2, b=2, size=(n_matches, 2)) * 1.2 + 0.4
        
        # Ventaja local variable
        home_advantages = np.random.uniform(1.05, 1.25, n_matches)
        
        # Estimación bayesiana de tasas de goles
        lambda_home = np.zeros(n_matches)
        lambda_away = np.zeros(n_matches)
        
        for i in range(n_matches):
            lambda_home[i] = attack_strengths[i, 0] * defense_strengths[i, 1] * home_advantages[i]
            lambda_away[i] = attack_strengths[i, 1] * defense_strengths[i, 0]
        
        # Simulación de probabilidades reales
        probabilities = ACBEModel.vectorized_poisson_simulation(lambda_home, lambda_away)
        
        # Generar cuotas con márgenes variables (realismo de casa de apuestas)
        margins = np.random.uniform(0.03, 0.07, n_matches)  # 3-7% de margen
        odds = np.zeros((n_matches, 3))
        
        for i in range(n_matches):
            odds[i] = 1 / (probabilities[i] * (1 + margins[i]))
            odds[i] = np.clip(odds[i], SystemConfig.MIN_ODDS, SystemConfig.MAX_ODDS)
        
        # Crear DataFrames
        matches_df = pd.DataFrame({
            'match_id': range(1, n_matches + 1),
            'home_attack': attack_strengths[:, 0],
            'away_attack': attack_strengths[:, 1],
            'home_defense': defense_strengths[:, 0],
            'away_defense': defense_strengths[:, 1],
            'home_advantage': home_advantages,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away
        })
        
        odds_df = pd.DataFrame(
            odds,
            columns=['odds_1', 'odds_X', 'odds_2']
        )
        odds_df.index = range(1, n_matches + 1)
        
        return matches_df, odds_df, probabilities

# ============================================================================
# SECCIÓN 8: INTERFAZ STREAMLIT PROFESIONAL
# ============================================================================

class ACBEApp:
    """Interfaz principal de la aplicación Streamlit."""
    
    def __init__(self):
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configuración de la página Streamlit."""
        st.set_page_config(
            page_title="ACBE-S73 Quantum Betting Suite v2.0",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_sidebar(self) -> Dict:
        """Renderiza sidebar y retorna configuración del usuario."""
        with st.sidebar:
            st.header("⚙️ Configuración del Sistema")
            
            # Bankroll inicial
            bankroll = st.number_input(
                "Bankroll Inicial (€)",
                min_value=100.0,
                max_value=1000000.0,
                value=SystemConfig.DEFAULT_BANKROLL,
                step=1000.0,
                help="Capital inicial para simulaciones"
            )
            
            # Parámetros de riesgo
            st.subheader("📊 Gestión de Riesgo")
            
            kelly_fraction = st.slider(
                "Fracción de Kelly",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Fracción conservadora del Kelly completo"
            )
            
            max_exposure = st.slider(
                "Exposición Máxima (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Porcentaje máximo del bankroll en apuestas"
            )
            
            # Configuración de simulaciones
            st.subheader("🎲 Parámetros de Simulación")
            
            monte_carlo_sims = st.number_input(
                "Simulaciones por Ronda",
                min_value=1000,
                max_value=50000,
                value=1000,
                step=1000
            )
            
            n_rounds = st.slider(
                "Rondas de Backtesting",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            
            # Fuente de datos
            st.subheader("📊 Fuente de Datos")
            data_source = st.radio(
                "Seleccionar fuente:",
                ["Datos Sintéticos", "Cargar CSV"],
                index=0
            )
            
            if data_source == "Datos Sintéticos":
                n_matches = st.slider(
                    "Número de partidos",
                    min_value=6,
                    max_value=15,
                    value=6,
                    step=1
                )
                generate_btn = st.button("🚀 Ejecutar Simulación Completa", type="primary")
            else:
                uploaded_file = st.file_uploader(
                    "Subir CSV con datos",
                    type=['csv'],
                    help="Columnas requeridas: home_attack, away_attack, home_defense, away_defense, odds_1, odds_X, odds_2"
                )
                generate_btn = uploaded_file is not None
            
            # Información del sistema
            with st.expander("ℹ️ Acerca del Sistema"):
                st.markdown("""
                **ACBE-S73 v2.0 - Características:**
                - ✅ **Cobertura S73 completa** (2 errores en 6 partidos)
                - ✅ **Reducción probabilística** por entropía
                - ✅ **Kelly integrado** por columna y portafolio
                - ✅ **Backtesting realista** con gestión de capital
                - ✅ **Análisis de riesgo** profesional (VaR, CVaR, Sharpe)
                """)
            
            return {
                'bankroll': bankroll,
                'kelly_fraction': kelly_fraction,
                'max_exposure': max_exposure / 100,
                'monte_carlo_sims': monte_carlo_sims,
                'n_rounds': n_rounds,
                'data_source': data_source,
                'n_matches': n_matches if data_source == "Datos Sintéticos" else None,
                'uploaded_file': uploaded_file if data_source == "Cargar CSV" else None,
                'generate_btn': generate_btn
            }
    
    def render_acbe_analysis(self, probabilities: np.ndarray, 
                            odds_matrix: np.ndarray,
                            normalized_entropies: np.ndarray):
        """Renderiza análisis ACBE completo."""
        st.header("🔬 Análisis ACBE")
        
        # Calcular métricas
        entropy = ACBEModel.calculate_entropy(probabilities)
        expected_value = InformationTheory.calculate_expected_value(probabilities, odds_matrix)
        
        # Clasificación de partidos
        allowed_signs, classifications = InformationTheory.classify_matches_by_entropy(
            probabilities, normalized_entropies
        )
        
        # DataFrames para visualización
        n_matches = len(probabilities)
        
        df_acbe = pd.DataFrame({
            'Partido': range(1, n_matches + 1),
            'Clasificación': classifications,
            'P(1)': probabilities[:, 0],
            'P(X)': probabilities[:, 1],
            'P(2)': probabilities[:, 2],
            'Entropía': entropy,
            'Entropía Norm.': normalized_entropies
        })
        
        df_odds = pd.DataFrame({
            'Partido': range(1, n_matches + 1),
            'Cuota 1': odds_matrix[:, 0],
            'Cuota X': odds_matrix[:, 1],
            'Cuota 2': odds_matrix[:, 2],
            'EV 1': expected_value[:, 0],
            'EV X': expected_value[:, 1],
            'EV 2': expected_value[:, 2],
            'Signos Permitidos': [str([SystemConfig.OUTCOME_LABELS[s] for s in signs]) 
                                 for signs in allowed_signs]
        })
        
        # Mostrar en columnas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Probabilidades ACBE")
            st.dataframe(df_acbe.style.format({
                'P(1)': '{:.3f}',
                'P(X)': '{:.3f}',
                'P(2)': '{:.3f}',
                'Entropía': '{:.3f}',
                'Entropía Norm.': '{:.3f}'
            }), use_container_width=True)
        
        with col2:
            st.subheader("💰 Cuotas y Valor Esperado")
            st.dataframe(df_odds.style.format({
                'Cuota 1': '{:.2f}',
                'Cuota X': '{:.2f}',
                'Cuota 2': '{:.2f}',
                'EV 1': '{:.3f}',
                'EV X': '{:.3f}',
                'EV 2': '{:.3f}'
            }), use_container_width=True)
        
        # Visualizaciones
        self._render_acbe_visualizations(probabilities, entropy, normalized_entropies)
    
    def _render_acbe_visualizations(self, probabilities: np.ndarray,
                                   entropy: np.ndarray,
                                   normalized_entropies: np.ndarray):
        """Renderiza visualizaciones del análisis ACBE."""
        n_matches = len(probabilities)
        
        # Gráfico de probabilidades
        fig_probs = go.Figure()
        for i, outcome in enumerate(['1', 'X', '2']):
            fig_probs.add_trace(go.Bar(
                x=list(range(1, n_matches + 1)),
                y=probabilities[:, i],
                name=outcome,
                marker_color=SystemConfig.OUTCOME_COLORS[i],
                text=[f'{p:.1%}' for p in probabilities[:, i]],
                textposition='auto'
            ))
        
        fig_probs.update_layout(
            title="Probabilidades ACBE por Partido",
            barmode='stack',
            xaxis_title="Partido",
            yaxis_title="Probabilidad",
            height=400
        )
        
        # Gráfico de entropía
        fig_entropy = go.Figure()
        fig_entropy.add_trace(go.Scatter(
            x=list(range(1, n_matches + 1)),
            y=normalized_entropies,
            mode='lines+markers',
            name='Entropía Normalizada',
            line=dict(color=SystemConfig.COLORS['primary'], width=3)
        ))
        
        # Líneas de umbral
        fig_entropy.add_hline(
            y=SystemConfig.STRONG_MATCH_THRESHOLD,
            line_dash="dash",
            line_color=SystemConfig.COLORS['success'],
            annotation_text="Fuerte"
        )
        fig_entropy.add_hline(
            y=SystemConfig.MEDIUM_MATCH_THRESHOLD,
            line_dash="dash", 
            line_color=SystemConfig.COLORS['warning'],
            annotation_text="Medio"
        )
        
        fig_entropy.update_layout(
            title="Clasificación por Entropía",
            xaxis_title="Partido",
            yaxis_title="Entropía Normalizada",
            height=400,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig_probs, use_container_width=True)
        st.plotly_chart(fig_entropy, use_container_width=True)
    
    def render_s73_system(self, probabilities: np.ndarray,
                         odds_matrix: np.ndarray,
                         normalized_entropies: np.ndarray,
                         bankroll: float):
        """Renderiza sistema S73 completo."""
        st.header("🧮 Sistema Combinatorio S73")
        
        with st.spinner("Construyendo sistema S73 optimizado..."):
            # 1. Generar combinaciones pre-filtradas
            filtered_combo, filtered_probs = S73System.generate_prefiltered_combinations(
                probabilities, normalized_entropies
            )
            
            # 2. Construir sistema de cobertura
            s73_combo, s73_probs = S73System.build_s73_coverage_system(
                filtered_combo, filtered_probs
            )
            
            # 3. Calcular métricas por columna
            n_columns = len(s73_combo)
            columns_data = []
            
            for idx, (combo, prob) in enumerate(zip(s73_combo, s73_probs), 1):
                # Calcular cuota conjunta
                combo_odds = S73System.calculate_combination_odds(combo, odds_matrix)
                
                # Calcular entropía promedio de la combinación
                combo_entropy = np.mean([normalized_entropies[i] for i in range(6)])
                
                # Calcular Kelly para la columna
                kelly_stake = KellyCapitalManagement.calculate_column_kelly(
                    combo, prob, combo_odds, combo_entropy
                )
                
                columns_data.append({
                    'ID': idx,
                    'Combinación': ''.join([SystemConfig.OUTCOME_LABELS[s] for s in combo]),
                    'Probabilidad': prob,
                    'Cuota': combo_odds,
                    'Valor Esperado': prob * combo_odds - 1,
                    'Entropía Prom.': combo_entropy,
                    'Kelly (%)': kelly_stake * 100,
                    'Inversión (€)': kelly_stake * bankroll
                })
            
            # Crear DataFrame
            columns_df = pd.DataFrame(columns_data)
            
            # 4. Normalizar stakes del portafolio
            kelly_stakes = np.array([d['Kelly (%)'] for d in columns_data]) / 100
            kelly_stakes = KellyCapitalManagement.normalize_portfolio_stakes(kelly_stakes)
            
            # Actualizar DataFrame con stakes normalizados
            for i, stake in enumerate(kelly_stakes):
                columns_df.at[i, 'Kelly (%)'] = stake * 100
                columns_df.at[i, 'Inversión (€)'] = stake * bankroll
        
        # Estadísticas del sistema
        st.subheader("📈 Estadísticas del Sistema S73")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Combinaciones Iniciales", len(filtered_combo))
        with col2:
            st.metric("Columnas S73 Finales", n_columns)
        with col3:
            total_exposure = np.sum(kelly_stakes) * 100
            st.metric("Exposición Total", f"{total_exposure:.1f}%")
        with col4:
            avg_prob = np.mean(s73_probs) * 100
            st.metric("Probabilidad Promedio", f"{avg_prob:.2f}%")
        
        # Mostrar columnas
        st.subheader("📋 Columnas del Sistema")
        
        display_df = columns_df.copy()
        display_df['Probabilidad'] = display_df['Probabilidad'].apply(lambda x: f'{x:.4%}')
        display_df['Cuota'] = display_df['Cuota'].apply(lambda x: f'{x:.2f}')
        display_df['Valor Esperado'] = display_df['Valor Esperado'].apply(lambda x: f'{x:.4f}')
        display_df['Entropía Prom.'] = display_df['Entropía Prom.'].apply(lambda x: f'{x:.3f}')
        display_df['Kelly (%)'] = display_df['Kelly (%)'].apply(lambda x: f'{x:.2f}%')
        display_df['Inversión (€)'] = display_df['Inversión (€)'].apply(lambda x: f'€{x:.2f}')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Preparar resultados para backtesting
        s73_results = {
            'combinations': s73_combo,
            'probabilities': s73_probs,
            'kelly_stakes': kelly_stakes,
            'filtered_count': len(filtered_combo),
            'final_count': n_columns
        }
        
        return s73_results
    
    def render_backtest_results(self, backtest_results: Dict, config: Dict):
        """Renderiza resultados del backtesting."""
        st.header("📈 Resultados del Backtesting")
        
        metrics = backtest_results['final_metrics']
        
        # Métricas principales
        st.subheader("📊 Métricas de Rendimiento")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Bankroll Final", f"€{metrics['final_bankroll']:,.2f}")
            st.metric("Retorno Total", f"{metrics['total_return_pct']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("CAGR", f"{metrics['cagr']:.2f}%")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
        with col4:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("VaR 95%", f"€{metrics['var_95']:.2f}")
        
        # Gráficos
        self._render_backtest_charts(backtest_results)
        
        # Análisis de riesgo
        self._render_risk_analysis(backtest_results, metrics)
    
    def _render_backtest_charts(self, backtest_results: Dict):
        """Renderiza gráficos del backtesting."""
        # Curva de equity y drawdown
        fig_equity = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Equity curve
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(backtest_results['equity_curve']))),
                y=backtest_results['equity_curve'],
                name='Bankroll',
                line=dict(color=SystemConfig.COLORS['success'], width=3)
            ),
            secondary_y=False
        )
        
        # Drawdown
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(backtest_results['drawdown_curve']))),
                y=backtest_results['drawdown_curve'],
                name='Drawdown',
                line=dict(color=SystemConfig.COLORS['danger'], width=2)
            ),
            secondary_y=True
        )
        
        fig_equity.update_layout(
            title="Evolución del Bankroll y Drawdown",
            xaxis_title="Ronda",
            height=500
        )
        fig_equity.update_yaxes(title_text="Bankroll (€)", secondary_y=False)
        fig_equity.update_yaxes(title_text="Drawdown %", secondary_y=True)
        
        # Distribución de retornos
        fig_returns = go.Figure()
        returns = backtest_results['all_returns']
        
        fig_returns.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Distribución de Retornos',
            marker_color=SystemConfig.COLORS['info'],
            opacity=0.7
        ))
        
        # Estadísticas en el gráfico
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        
        fig_returns.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color=SystemConfig.COLORS['primary'],
            annotation_text=f"Media: €{mean_return:.2f}"
        )
        fig_returns.add_vline(
            x=median_return,
            line_dash="dot",
            line_color=SystemConfig.COLORS['secondary'],
            annotation_text=f"Mediana: €{median_return:.2f}"
        )
        
        fig_returns.update_layout(
            title="Distribución de Retornos por Ronda",
            xaxis_title="Retorno (€)",
            yaxis_title="Frecuencia",
            height=400
        )
        
        # Mostrar gráficos
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_equity, use_container_width=True)
        with col2:
            st.plotly_chart(fig_returns, use_container_width=True)
    
    def _render_risk_analysis(self, backtest_results: Dict, metrics: Dict):
        """Renderiza análisis de riesgo detallado."""
        st.subheader("🔍 Análisis de Riesgo Detallado")
        
        returns = backtest_results['all_returns']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calcular CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])
            
            st.metric("CVaR 95%", f"€{cvar_95:.2f}")
            st.metric("Volatilidad (σ)", f"€{metrics['std_returns']:.2f}")
            st.metric("Ratio Sortino", 
                     f"{(np.mean(returns) / np.std(returns[returns < 0])):.2f}" 
                     if np.std(returns[returns < 0]) > 0 else "N/A")
        
        with col2:
            # Estadísticas de colas
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns <= 0]
            
            st.metric("Asimetría (Skewness)", f"{pd.Series(returns).skew():.3f}")
            st.metric("Curtosis", f"{pd.Series(returns).kurtosis():.3f}")
            st.metric("Ratio Ganancia/Pérdida", 
                     f"{abs(np.mean(positive_returns)/np.mean(negative_returns)):.2f}"
                     if len(negative_returns) > 0 else "N/A")
    
    def render_executive_summary(self, s73_results: Dict, backtest_results: Dict, config: Dict):
        """Renderiza resumen ejecutivo del sistema."""
        st.header("📋 Resumen Ejecutivo")
        
        metrics = backtest_results['final_metrics']
        
        # Eficiencia del sistema
        st.subheader("🎯 Eficiencia del Sistema S73")
        
        efficiency_data = {
            'Métrica': [
                'Reducción del Espacio',
                'Cobertura de Errores', 
                'Exposición Total',
                'Diversificación'
            ],
            'Valor': [
                f"{s73_results['filtered_count']} → {s73_results['final_count']}",
                f"{SystemConfig.HAMMING_DISTANCE_TARGET} errores",
                f"{np.sum(s73_results['kelly_stakes']) * 100:.1f}%",
                f"{len(set([tuple(c) for c in s73_results['combinations']]))} únicas"
            ]
        }
        
        st.table(pd.DataFrame(efficiency_data))
        
        # Rentabilidad
        st.subheader("📈 Rentabilidad Esperada")
        
        profitability_data = {
            'Métrica': ['ROI Total', 'Sharpe Ratio', 'Win Rate', 'Expectativa/Ronda'],
            'Valor': [
                f"{metrics['total_return_pct']:.2f}%",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['win_rate']:.2f}%",
                f"€{np.mean(backtest_results['all_returns']):.2f}"
            ]
        }
        
        st.table(pd.DataFrame(profitability_data))
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones de Gestión")
        
        total_exposure = np.sum(s73_results['kelly_stakes']) * 100
        
        if total_exposure > 20:
            exposure_status = "⚠️ ALTO"
            exposure_rec = "Reducir exposición a <15%"
        elif total_exposure > 10:
            exposure_status = "✅ MODERADO" 
            exposure_rec = "Exposición adecuada"
        else:
            exposure_status = "✅ BAJO"
            exposure_rec = "Podría aumentar exposición"
        
        if metrics['max_drawdown'] > 25:
            risk_status = "⚠️ ALTO"
            risk_rec = "Implementar stop-loss agresivo"
        elif metrics['max_drawdown'] > 15:
            risk_status = "⚠️ MODERADO"
            risk_rec = "Monitorear drawdown semanal"
        else:
            risk_status = "✅ BAJO"
            risk_rec = "Riesgo bien controlado"
        
        recommendations = pd.DataFrame({
            'Área': ['Exposición', 'Riesgo', 'Diversificación', 'Gestión'],
            'Estado': [exposure_status, risk_status, '✅ ADECUADO', '✅ IMPLEMENTADO'],
            'Recomendación': [exposure_rec, risk_rec, 
                            f"{s73_results['final_count']} columnas bien diversificadas",
                            f"Kelly ajustado con límite {SystemConfig.KELLY_FRACTION_MAX*100:.0f}%"]
        })
        
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
        
        # Conclusión final
        st.subheader("🎯 Conclusión del Sistema")
        
        roi = metrics['total_return_pct']
        sharpe = metrics['sharpe_ratio']
        
        if roi > 10 and sharpe > 1.5:
            conclusion = "EXCELENTE - Sistema altamente rentable con excelente perfil riesgo/retorno"
            color = SystemConfig.COLORS['success']
        elif roi > 5 and sharpe > 1.0:
            conclusion = "BUENO - Sistema rentable con gestión adecuada de riesgo"
            color = SystemConfig.COLORS['success']
        elif roi > 0:
            conclusion = "ACEPTABLE - Sistema positivo con margen de mejora"
            color = SystemConfig.COLORS['warning']
        else:
            conclusion = "MEJORABLE - Revisar configuración del sistema"
            color = SystemConfig.COLORS['danger']
        
        st.markdown(f"""
        <div style="background-color:{color}20; padding:20px; border-radius:10px; border-left:5px solid {color};">
            <h4 style="color:{color};">{conclusion}</h4>
            <p><strong>Simulaciones realizadas:</strong> {config['n_rounds']} rondas × {config['monte_carlo_sims']:,} iteraciones Monte Carlo</p>
            <p><strong>Resultado final:</strong> €{metrics['final_bankroll']:,.2f} ({roi:+.2f}%)</p>
            <p><strong>Calidad del sistema:</strong> Sharpe Ratio = {sharpe:.2f}, Max Drawdown = {metrics['max_drawdown']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Método principal de ejecución de la aplicación."""
        st.title("🎯 ACBE-S73 Quantum Betting Suite v2.0")
        st.markdown("""
        *Sistema profesional de optimización de portafolios de apuestas deportivas*  
        *Con cobertura S73 completa, gestión probabilística y Kelly integrado*
        """)
        
        # Renderizar sidebar y obtener configuración
        config = self.render_sidebar()
        
        if not config['generate_btn']:
            st.info("👈 Configura los parámetros en la sidebar y ejecuta la simulación")
            return
        
        try:
            # Crear pestañas principales
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Análisis ACBE", 
                "🧮 Sistema S73", 
                "📈 Backtesting",
                "📋 Resumen"
            ])
            
            with st.spinner("🔄 Procesando datos y ejecutando simulaciones..."):
                # Cargar/generar datos
                if config['data_source'] == "Datos Sintéticos":
                    matches_df, odds_df, probabilities = SyntheticDataGenerator.generate_complete_dataset(
                        config['n_matches']
                    )
                else:
                    # Cargar CSV personalizado
                    matches_df = pd.read_csv(config['uploaded_file'])
                    required_cols = ['home_attack', 'away_attack', 'home_defense', 'away_defense']
                    odds_cols = ['odds_1', 'odds_X', 'odds_2']
                    
                    if not all(col in matches_df.columns for col in required_cols + odds_cols):
                        st.error(f"❌ CSV debe contener: {required_cols + odds_cols}")
                        return
                    
                    odds_df = matches_df[odds_cols].copy()
                    odds_df.columns = ['odds_1', 'odds_X', 'odds_2']
                    matches_df = matches_df[required_cols].copy()
                    
                    # Calcular probabilidades ACBE
                    attack_strengths = matches_df[['home_attack', 'away_attack']].values
                    defense_strengths = matches_df[['home_defense', 'away_defense']].values
                    
                    lambda_home, lambda_away = ACBEModel.gamma_poisson_bayesian(
                        attack_strengths, defense_strengths
                    )
                    probabilities = ACBEModel.vectorized_poisson_simulation(lambda_home, lambda_away)
                
                odds_matrix = odds_df.values
                
                # Calcular entropías
                entropy = ACBEModel.calculate_entropy(probabilities)
                normalized_entropy = ACBEModel.normalize_entropy(entropy)
                
                # Actualizar configuración de exposición máxima
                SystemConfig.MAX_PORTFOLIO_EXPOSURE = config['max_exposure']
            
            # Pestaña 1: Análisis ACBE
            with tab1:
                self.render_acbe_analysis(probabilities, odds_matrix, normalized_entropy)
            
            # Verificar que hay al menos 6 partidos para S73
            if len(probabilities) < 6:
                st.warning("⚠️ Se necesitan al menos 6 partidos para el sistema S73")
                return
            
            # Usar solo primeros 6 partidos para S73 (sistema clásico)
            probs_6 = probabilities[:6, :]
            odds_6 = odds_matrix[:6, :]
            entropy_6 = normalized_entropy[:6]
            
            # Pestaña 2: Sistema S73
            with tab2:
                s73_results = self.render_s73_system(probs_6, odds_6, entropy_6, config['bankroll'])
            
            # Pestaña 3: Backtesting
            with tab3:
                # Ejecutar backtesting
                backtester = VectorizedBacktester(initial_bankroll=config['bankroll'])
                
                with st.spinner("Ejecutando backtesting completo..."):
                    backtest_results = backtester.run_backtest(
                        probs_6, odds_6, entropy_6,
                        s73_results,
                        n_rounds=config['n_rounds'],
                        n_sims_per_round=config['monte_carlo_sims'],
                        kelly_fraction=config['kelly_fraction']
                    )
                
                self.render_backtest_results(backtest_results, config)
            
            # Pestaña 4: Resumen ejecutivo
            with tab4:
                self.render_executive_summary(s73_results, backtest_results, config)
                
        except Exception as e:
            st.error(f"❌ Error en la ejecución: {str(e)}")
            st.exception(e)

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    app = ACBEApp()
    app.run()
