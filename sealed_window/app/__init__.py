"""The web application (FastAPI + a thin static front end), served in the SEALED process role.

Modules
-------
server         :func:`create_app` -- JSON API for snapshots, slider schema, plan compilation and
               approval, runs, the seal indicator, the claim ledger and dossiers.
static/        ``index.html``: left rail with Fundamental and Technical slider tabs, the seal
               indicator, spend approval and the claim ledger.

The app has no endpoint that acquires data, and the process it runs in cannot import the
Upstox adapter. Slider values travel only to the deterministic screen.
"""
