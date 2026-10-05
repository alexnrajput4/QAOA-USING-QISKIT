"""Streamlit demo: quantum portfolio picker (QAOA).   Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st

import viz
from portfolio_qaoa import (QAOA, bitstring, int_to_bits, make_problem, portfolio_stats,
                            problem_from_prices, solve)

try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

st.set_page_config(page_title="Quantum Portfolio Picker", layout="wide")
st.title("Quantum portfolio picker (QAOA)")
st.caption("Choose exactly K of N assets that balance expected return against risk. "
           "A QAOA circuit (Qiskit) proposes portfolios; brute force and greedy search are the baselines.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Problem")
    source = st.radio("Data", ["Synthetic demo data", "Upload prices CSV"])
    names, data = None, None
    if source == "Upload prices CSV":
        up = st.file_uploader("CSV: one column per asset, one row per day (closing prices)", type="csv")
        if up is not None:
            df = pd.read_csv(up).select_dtypes("number").dropna()
            df = df.iloc[:, :10]
            if df.shape[1] >= 3 and len(df) > 20:
                names = list(df.columns)
                data = problem_from_prices(df.to_numpy())
            else:
                st.warning("Need at least 3 numeric columns and 20 rows.")
        n = len(names) if names else 8
        if names:
            st.caption(f"Using {n} assets (max 10).")
    else:
        n = st.slider("Number of assets (qubits)", 4, 10, 8)
    k = st.slider("Assets to pick (K)", 1, max(n - 1, 1), min(max(n // 2, 1), n - 1))
    risk = st.slider("Risk aversion", 0.1, 2.0, 0.5, 0.1)
    seed = st.number_input("Random seed (synthetic data)", 0, 999, 2)

    st.header("Quantum settings")
    p = st.slider("QAOA depth p (layers)", 1, 5, 3)
    mixer = st.radio("Mixer", ["xy", "x"], format_func=lambda m: {
        "xy": "XY ring (always exactly K assets)", "x": "X mixer + penalty (can break the budget)"}[m])
    alpha = st.slider("CVaR alpha", 0.05, 0.5, 0.10, 0.05,
                      help="Optimise the average of the best alpha-fraction of shots.")
    shots = st.select_slider("Shots", [256, 512, 1024, 2048, 4096, 8192], 2048)
    use_qiskit = st.checkbox("Evaluate the final circuit with Qiskit", value=HAS_QISKIT,
                             disabled=not HAS_QISKIT)
    run = st.button("Run", type="primary", use_container_width=True)
    if n >= 10 and p >= 4:
        st.info("n=10, p>=4 can take 20 s or more.")

names = names or [f"A{i + 1}" for i in range(n)]
params = (source, n, k, risk, seed, p, mixer, alpha, shots, use_qiskit, tuple(names),
          None if data is None else float(np.sum(data[0])))

if run or "result" not in st.session_state:
    with st.spinner("Tuning the circuit angles..."):
        st.session_state["result"] = solve(
            n=n, k=k, risk=risk, p=p, shots=shots, seed=int(seed), data=data, mixer=mixer, alpha=alpha,
            backend="qiskit" if (use_qiskit and HAS_QISKIT) else "numpy")
        st.session_state["params"] = params
        st.session_state["names"] = names

r = st.session_state["result"]
if st.session_state.get("params") != params:
    st.info("Settings changed - press Run to update the results.")
names = st.session_state["names"]
q, n = r["qubo"], r["qubo"].n


def pick(b):
    return ", ".join(nm for nm, bit in zip(names, int_to_bits(b, n)) if bit)


# ------------------------------------------------------------------ headline metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid portfolios in QAOA output", f"{r['p_feasible'] * 100:.0f}%",
          f"random bitstring: {r['p_random_feasible'] * 100:.0f}%", delta_color="off")
found = r["qaoa_best"] == r["optimal"]
c2.metric("Optimum found by QAOA", "Yes" if found else "No",
          None if found else f"gap {r['gap_qaoa'] * 100:.1f}%", delta_color="off")
c3.metric("Probability on best 5 portfolios", f"{r['p_top5'] * 100:.0f}%",
          f"random: {r['p_random_top5'] * 100:.0f}%", delta_color="off")
c4.metric("Greedy gap to optimum", f"{r['gap_greedy'] * 100:.1f}%", delta_color="off")

# ------------------------------------------------------------------ comparison table
rows = []
for label, b, e in (("Brute force (optimal)", r["optimal"], r["optimal_energy"]),
                    ("Greedy", r["greedy"], r["greedy_energy"]),
                    ("QAOA (best of shots)", r["qaoa_best"], r["qaoa_best_energy"])):
    if b is None:
        rows.append({"Method": label, "Assets": "no valid sample", "Return %": None, "Risk %": None,
                     "Objective": None})
        continue
    s = portfolio_stats(int_to_bits(b, n), q.mu, q.sigma)
    rows.append({"Method": label, "Assets": pick(b), "Return %": round(s["return"] * 100, 2),
                 "Risk %": round(s["risk"] * 100, 2), "Objective": round(float(e), 4)})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("Objective = risk_aversion * variance - expected return (lower is better), equal weights on the "
           "K chosen assets.")

# ------------------------------------------------------------------ charts
t1, t2, t3, t4 = st.tabs(["Solution map", "QAOA output", "Circuit", "About & honest caveats"])
with t1:
    st.pyplot(viz.fig_frontier(r))
with t2:
    a, b = st.columns(2)
    a.pyplot(viz.fig_distribution(r))
    b.pyplot(viz.fig_convergence(r))
with t3:
    if HAS_QISKIT and r["qaoa"].backend == "qiskit":
        try:
            diff = r["qaoa"].verify_with_qiskit(r["result"].params)
            st.success(f"Qiskit circuit vs fast numpy engine: max probability difference {diff:.1e}")
        except Exception as exc:  # keep the demo alive
            st.warning(f"Cross-check unavailable: {exc}")
        qc = r["qaoa"].circuit
        st.write(f"Full circuit (p={r['qaoa'].p}): {qc.num_qubits} qubits, depth {qc.depth()}, "
                 f"gates {dict(qc.count_ops())}")
        try:
            one = QAOA(r["qubo"], p=1, backend="qiskit", mixer=mixer, alpha=alpha).circuit
            st.code(str(one.draw("text", fold=140)), language=None)
            st.caption("One QAOA layer shown. Start state, then cost layer (RZ, RZZ), then mixer.")
        except Exception as exc:
            st.warning(f"Could not draw circuit: {exc}")
    else:
        st.info("Install qiskit and tick 'Evaluate the final circuit with Qiskit' to see the circuit.")
with t4:
    st.markdown("""
**What happens**
1. Portfolio choice becomes a QUBO: minimise `risk * x^T Sigma x - mu^T x` over bitstrings `x` with exactly K ones.
2. The QUBO maps to an Ising Hamiltonian; each layer of QAOA applies the cost phases, then a mixer.
3. The **XY ring mixer** keeps the number of selected assets fixed, so every sample is a valid portfolio.
4. A classical optimiser tunes the angles on a **CVaR** objective (average of the best samples).
5. We sample shots and keep the lowest-energy valid portfolio.

**Honest caveats**
- This is a classical simulation of a small circuit. Brute force over a few hundred portfolios is instant,
  so there is **no quantum advantage** here; the point is a working, verifiable hybrid pipeline.
- The default synthetic instance (seed 2) is one where the greedy baseline is not optimal; results vary by
  instance and QAOA gets harder as N grows (try N=10 with p=3).
- Equal weights only; real portfolio optimisation also needs weights, transaction costs, constraints.
- The next step would be running the same circuit on IBM hardware with `qiskit-ibm-runtime`.
""")
