"""Phase 6 -- PUBLISH. Model: none. Network: sealed.

The dossier is assembled deterministically from surviving claims. There is no generative step:
every sentence traces to a claim, every claim to evidence IDs, every evidence ID to a row in a
hashed snapshot.

Modules
-------
dossier  Recommendation rules (BUY / WATCH / AVOID / abstain), the dossier structure with the
         reproducibility triple, and a Markdown renderer.
"""
