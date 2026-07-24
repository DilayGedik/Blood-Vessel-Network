# Blood Vessel Network Perfusion Simulator

Interactive branching vascular network model with pressure, flow, velocity,
wall shear stress, tracer transport, and occlusion analysis.

## Core equations
    R = 8 μ L / (π r⁴)
    Q = ΔP / R

Internal pressures are solved by enforcing flow conservation at junctions.

## Run
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

On macOS/Linux:
    source .venv/bin/activate

## Scope
Reduced order portfolio model. Not a validated clinical or patient specific CFD tool.
