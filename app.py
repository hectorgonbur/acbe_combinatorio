# ============================================
# ACBE-S73 Manual Institutional v2.0
# Parte 1: Arquitectura Base + UI Manual
# ============================================

import streamlit as st
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict
from itertools import product

# ============================================
# CONFIGURACIÓN GLOBAL INSTITUCIONAL
# ============================================

MAX_MATCHES = 6
MAX_KELLY = 0.03
MAX_PORTFOLIO_EXPOSURE = 0.15

ALPHA_S73 = 0.25


# ============================================
# MODELOS DE DATOS
# ============================================

@dataclass
class MatchInput:
    league: str
    home_team: str
    away_team: str
    odd_1: float
    odd_x: float
    odd_2: float


@dataclass
class ManualInputData:
    bankroll: float
    kelly_fraction: float
    matches: List[MatchInput]


# ============================================
# APLICACIÓN PRINCIPAL
# ============================================

class ACBEApp:

    def __init__(self):
        self.initialize_session_state()
        self.acbe_module = ACBEModule()
        self.s73_module = S73Module()
        self.portfolio_manager = PortfolioManager(
            max_kelly=MAX_KELLY,
            max_exposure=MAX_PORTFOLIO_EXPOSURE
        )
        self.montecarloengine =  MonteCarloEngine()


    # ----------------------------------------
    # Inicialización segura del session_state
    # ----------------------------------------
    def initialize_session_state(self):
        if "input_data" not in st.session_state:
            st.session_state["input_data"] = None

    # ----------------------------------------
    # Render principal
    # ----------------------------------------
    def run(self):
        st.set_page_config(page_title="ACBE-S73 Manual Institutional v2.0", layout="wide")
        st.title("ACBE-S73 Manual Institutional v2.0")
        st.markdown("Sistema Cuantitativo Institucional - Modo Manual Exclusivo")

        self.render_manual_form()
        self.process_pipeline()
        self.render_results()
        self.render_monte_carlo()

    # ----------------------------------------
    # Formulario Manual Completo
    # ----------------------------------------
    def render_manual_form(self):

        with st.form("manual_input_form"):

            st.header("Configuración de Capital")

            bankroll = st.number_input(
                "Bankroll Inicial",
                min_value=0.0,
                step=1000.0,
                format="%.2f"
            )

            kelly_fraction = st.slider(
                "Fracción de Kelly",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )

            st.divider()
            st.header("Carga Manual de Partidos")

            matches = []

            for i in range(MAX_MATCHES):

                st.subheader(f"Partido {i+1}")

                col1, col2 = st.columns(2)

                with col1:
                    league = st.text_input(f"Liga {i+1}", key=f"league_{i}")
                    home_team = st.text_input(f"Equipo Local {i+1}", key=f"home_{i}")
                    odd_1 = st.number_input(
                        f"Cuota 1 - Partido {i+1}",
                        min_value=1.01,
                        step=0.01,
                        key=f"odd1_{i}"
                    )

                with col2:
                    away_team = st.text_input(f"Equipo Visitante {i+1}", key=f"away_{i}")
                    odd_x = st.number_input(
                        f"Cuota X - Partido {i+1}",
                        min_value=1.01,
                        step=0.01,
                        key=f"oddx_{i}"
                    )
                    odd_2 = st.number_input(
                        f"Cuota 2 - Partido {i+1}",
                        min_value=1.01,
                        step=0.01,
                        key=f"odd2_{i}"
                    )

                matches.append({
                    "league": league,
                    "home_team": home_team,
                    "away_team": away_team,
                    "odd_1": odd_1,
                    "odd_x": odd_x,
                    "odd_2": odd_2
                })

            submitted = st.form_submit_button("Ejecutar Análisis")

        if submitted:
            st.session_state["input_data"] = {
                "bankroll": bankroll,
                "kelly_fraction": kelly_fraction,
                "matches": matches
            }


    def process_pipeline(self):

        if "input_data" not in st.session_state:
            return

        input_data = st.session_state["input_data"]

        # ACBE
        acbe_results = self.acbe_module.process_matches(input_data)
        st.session_state["acbe_results"] = acbe_results

        # S73
        s73_results = self.s73_module.build_s73(acbe_results, input_data)
        st.session_state["s73_results"] = s73_results

        # Portfolio
        portfolio_results = self.portfolio_manager.allocate_portfolio(
            acbe_results=acbe_results,
            s73_results=s73_results,
            input_data=input_data,
            bankroll=input_data["bankroll"],
            kelly_fraction_user=input_data["kelly_fraction"]
        )

        st.session_state["portfolio_results"] = portfolio_results


        # ----------------------------------------
        # Confirmación de persistencia
        # ----------------------------------------

    def render_results(self):

        if "portfolio_results" not in st.session_state:
            return

        st.header("Resultados Institucionales")

        # ACBE
        if "acbe_results" in st.session_state:

            results = st.session_state["acbe_results"]

            st.subheader("ACBE")

            for i in range(len(results["probabilities"])):

                st.markdown(f"### Partido {i+1}")
                st.json(results["probabilities"][i])
                st.write("Entropía:", round(results["entropy"][i], 4))
                st.write("Clasificación:", results["classification"][i])
                st.json(results["ev_matrix"][i])
                st.divider()

        # S73
        if "s73_results" in st.session_state:

            st.subheader("Sistema S73")

            for i, col in enumerate(st.session_state["s73_results"]["columns"]):

                st.write(
                    f"Columna {i+1}: {col['column']} | "
                    f"P={round(col['joint_probability'],6)} | "
                    f"Odds={round(col['joint_odds'],3)} | "
                    f"EV={round(col['joint_ev'],6)}"
                )

        # Portfolio
        portfolio = st.session_state["portfolio_results"]

        st.subheader("Portafolio")

        for bet in portfolio["portfolio"]:

            if bet["type"] == "single":
                st.write(
                    f"Single M{bet['match_index']} {bet['selection']} | "
                    f"Stake={round(bet['stake'],2)}"
                )
            else:
                st.write(
                    f"S73 {bet['column']} | "
                    f"Stake={round(bet['stake'],2)}"
                )

        st.write("Exposición total:", round(portfolio["total_exposure"], 2))
        st.write("ROI esperado:", round(portfolio["roi_expected"], 6))


    def render_monte_carlo(self):

        if "portfolio_results" not in st.session_state:
            return

        st.markdown("### 🔬 Simulación Monte Carlo")

        if st.button("Ejecutar Simulación Monte Carlo"):

            portfolio = st.session_state["portfolio_results"]["portfolio"]
            bankroll = st.session_state["input_data"]["bankroll"]

            mc_engine = MonteCarloEngine(simulations=10000)

            st.session_state["mc_results"] = mc_engine.simulate_portfolio(
                portfolio,
                bankroll
            )

        if "mc_results" in st.session_state:

            mc = st.session_state["mc_results"]

            st.write("ROI Medio:", round(mc["roi_mean"], 4))
            st.write("Desvío ROI:", round(mc["roi_std"], 4))
            st.write("Prob ROI positivo:", round(mc["prob_positive"], 4))
            st.write("Drawdown Medio:", round(mc["max_dd_mean"], 4))
            st.write("DD 95%:", round(mc["dd_95"], 4))

# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    app = ACBEApp()
    app.run()

# ============================================
# MOTOR MATEMÁTICO ACBE
# ============================================

import math


class ACBEModule:

    def __init__(self):
        pass

    def process_matches(self, input_data: dict) -> dict:

        probabilities = []
        entropy_list = []
        classification = []
        ev_matrix = []

        matches = input_data["matches"]

        for match in matches:

            odd_1 = match["odd_1"]
            odd_x = match["odd_x"]
            odd_2 = match["odd_2"]

            # ----------------------------------------
            # 1️⃣ Probabilidades implícitas
            # ----------------------------------------
            p1 = 1 / odd_1
            px = 1 / odd_x
            p2 = 1 / odd_2

            prob_sum = p1 + px + p2

            # ----------------------------------------
            # 2️⃣ Eliminación del margen bookmaker
            # ----------------------------------------
            p1_norm = p1 / prob_sum
            px_norm = px / prob_sum
            p2_norm = p2 / prob_sum

            probabilities.append({
                "1": p1_norm,
                "X": px_norm,
                "2": p2_norm
            })

            # ----------------------------------------
            # 3️⃣ Entropía Shannon base 3
            # ----------------------------------------
            entropy = 0
            for p in [p1_norm, px_norm, p2_norm]:
                if p > 0:
                    entropy += -p * math.log(p)

            entropy = entropy / math.log(3)

            entropy_list.append(entropy)

            # ----------------------------------------
            # 4️⃣ Clasificación estructural
            # ----------------------------------------
            if entropy <= 0.45:
                cls = "Fuerte"
            elif entropy <= 0.75:
                cls = "Medio"
            else:
                cls = "Caótico"


            classification.append(cls)

            # ----------------------------------------
            # 5️⃣ Expected Value por signo
            # ----------------------------------------
            ev_1 = p1_norm * odd_1 - 1
            ev_x = px_norm * odd_x - 1
            ev_2 = p2_norm * odd_2 - 1

            ev_matrix.append({
                "1": ev_1,
                "X": ev_x,
                "2": ev_2
            })

        return {
            "probabilities": probabilities,
            "entropy": entropy_list,
            "classification": classification,
            "ev_matrix": ev_matrix
        }

# ============================================
# SISTEMA S73 ÓPTIMO INSTITUCIONAL
# ============================================

class S73Module:

    def __init__(self, max_columns=73, alpha=0.25):
        self.max_columns = max_columns
        self.alpha = alpha
        self.sign_map = {0: "1", 1: "X", 2: "2"}

    # -------------------------------------------------
    # Distancia Hamming
    # -------------------------------------------------
    def hamming_distance(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    # -------------------------------------------------
    # Construcción S73 Óptimo
    # -------------------------------------------------
    def build_s73(self, acbe_results, input_data):

        probabilities = acbe_results["probabilities"]
        classifications = acbe_results["classification"]

        allowed_space = []

        # 1️⃣ Determinar espacio permitido por partido
        for i, probs in enumerate(probabilities):

            # ordenar signos por prob descendente
            sorted_signs = sorted(
                probs.items(),
                key=lambda x: x[1],
                reverse=True
            )

            if classifications[i] == "Fuerte":
                allowed = [sorted_signs[0][0]]

            elif classifications[i] == "Medio":
                allowed = [sorted_signs[0][0], sorted_signs[1][0]]

            else:  # Caótico
                allowed = ["1", "X", "2"]

            allowed_space.append(allowed)

        # 2️⃣ Producto cartesiano
        all_combos = list(product(*allowed_space))

        columns_data = []

        # 3️⃣ Calcular métricas conjuntas
        for combo in all_combos:

            joint_prob = 1.0
            joint_odds = 1.0

            for i, sign in enumerate(combo):

                # Probabilidad
                p = probabilities[i][sign]
                joint_prob *= p

                # Cuota correcta
                match_data = input_data["matches"][i]

                if sign == "1":
                    odd = match_data["odd_1"]
                elif sign == "X":
                    odd = match_data["odd_x"]
                else:
                    odd = match_data["odd_2"]

                joint_odds *= odd

            joint_ev = joint_prob * joint_odds - 1

            # Score institucional
            if joint_prob > 0:
                score = math.log(joint_prob) + self.alpha * joint_ev
            else:
                score = -9999

            columns_data.append({
                "column": combo,
                "joint_probability": joint_prob,
                "joint_odds": joint_odds,
                "joint_ev": joint_ev,
                "score": score
            })

        # 4️⃣ Ordenar por score descendente
        columns_data.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # 5️⃣ Aplicar filtro Hamming ≥ 2
        selected = []

        for candidate in columns_data:

            if len(selected) >= self.max_columns:
                break

            if all(
                self.hamming_distance(candidate["column"], s["column"]) >= 2
                for s in selected
            ):
                selected.append(candidate)

        return {
            "columns": selected
        }

# ============================================================
# PORTFOLIO MANAGER – NIVEL 1 INSTITUCIONAL
# ============================================================

class PortfolioManager:

    def __init__(self, max_kelly=0.03, max_exposure=0.15):
        self.max_kelly = max_kelly
        self.max_exposure = max_exposure

    # =====================================================
    # Kelly robusto con penalización por entropía
    # =====================================================
    def kelly_fraction(self, p, q, entropy, kelly_fraction_user):

        # Kelly bruto
        numerator = p * q - 1
        denominator = q - 1

        if denominator <= 0:
            return 0.0

        f = numerator / denominator

        # No apuestas negativas
        if f <= 0:
            return 0.0

        # Aplicar fracción usuario
        f *= kelly_fraction_user

        # Límite institucional
        f = min(f, self.max_kelly)

        # Penalización por entropía
        f *= (1 - entropy)

        return max(0.0, f)

    # =====================================================
    # Asignación de portafolio completa
    # =====================================================
    def allocate_portfolio(self, acbe_results, s73_results,
                           input_data, bankroll, kelly_fraction_user):

        probabilities = acbe_results["probabilities"]
        entropies = acbe_results["entropy"]
        ev_matrix = acbe_results["ev_matrix"]

        portfolio = []

        # =====================================================
        # 1️⃣ SINGLES
        # =====================================================
        for i in range(len(probabilities)):

            for sign in ["1", "X", "2"]:

                p = probabilities[i][sign]
                ev = ev_matrix[i][sign]

                if ev <= 0:
                    continue

                # Obtener cuota real del input
                match = input_data["matches"][i]
                if sign == "1":
                    q = match["odd_1"]
                elif sign == "X":
                    q = match["odd_x"]
                else:
                    q = match["odd_2"]

                entropy = entropies[i]

                f = self.kelly_fraction(p, q, entropy, kelly_fraction_user)

                if f > 0:
                    portfolio.append({
                        "type": "single",
                        "match_index": i,
                        "selection": sign,
                        "probability": p,
                        "odds": q,
                        "ev": ev,
                        "entropy": entropy,
                        "stake_fraction": f
                    })

        # =====================================================
        # 2️⃣ COLUMNAS S73
        # =====================================================
        for col in s73_results["columns"]:

            joint_prob = col["joint_probability"]
            joint_odds = col["joint_odds"]
            joint_ev = col["joint_ev"]
            combo = col["column"]

            if joint_ev <= 0:
                continue

            # Entropía promedio de la columna
            column_entropy = sum(entropies[i] for i in range(len(combo))) / len(combo)

            f = self.kelly_fraction(
                joint_prob,
                joint_odds,
                column_entropy,
                kelly_fraction_user
            )

            if f > 0:
                portfolio.append({
                    "type": "column",
                    "column": combo,
                    "probability": joint_prob,
                    "odds": joint_odds,
                    "ev": joint_ev,
                    "entropy": column_entropy,
                    "stake_fraction": f
                })

        # =====================================================
        # 3️⃣ CONTROL EXPOSICIÓN TOTAL (PROPORCIONAL)
        # =====================================================
        total_fraction = sum(bet["stake_fraction"] for bet in portfolio)

        if total_fraction > self.max_exposure and total_fraction > 0:

            scale = self.max_exposure / total_fraction

            for bet in portfolio:
                bet["stake_fraction"] *= scale

            total_fraction = self.max_exposure

        # =====================================================
        # 4️⃣ CALCULAR STAKES EN DINERO
        # =====================================================
        for bet in portfolio:
            bet["stake"] = bet["stake_fraction"] * bankroll

        total_exposure = sum(bet["stake"] for bet in portfolio)

        # =====================================================
        # 5️⃣ ROI ESPERADO AGREGADO
        # =====================================================
        roi_expected = sum(
            bet["stake_fraction"] * bet["ev"]
            for bet in portfolio
        )

        return {
            "portfolio": portfolio,
            "total_exposure": total_exposure,
            "total_fraction": total_fraction,
            "roi_expected": roi_expected
        }

class MonteCarloEngine:

    def __init__(self, simulations=10000):
        self.simulations = simulations

    def simulate_portfolio(self, portfolio, bankroll):

        results = []
        drawdowns = []

        for _ in range(self.simulations):

            current_bankroll = bankroll
            peak = bankroll
            max_dd = 0

            for bet in portfolio:

                p = bet["probability"]
                stake = bet["stake"]
                odds = bet["odds"]

                outcome = np.random.rand()

                if outcome <= p:
                    profit = stake * (odds - 1)
                else:
                    profit = -stake

                current_bankroll += profit

                if current_bankroll > peak:
                    peak = current_bankroll

                dd = (peak - current_bankroll) / peak
                max_dd = max(max_dd, dd)

            roi = (current_bankroll - bankroll) / bankroll

            results.append(roi)
            drawdowns.append(max_dd)

        return {
            "roi_mean": np.mean(results),
            "roi_std": np.std(results),
            "prob_positive": np.mean(np.array(results) > 0),
            "max_dd_mean": np.mean(drawdowns),
            "dd_95": np.percentile(drawdowns, 95),
            "roi_5": np.percentile(results, 5),
            "roi_95": np.percentile(results, 95)
        }
