"""Learner library. Importing registers every implemented learner in REGISTRY; methods named in the coverage
matrix but not yet implemented are listed in PLANNED with their provenance."""
from .base import PLANNED, REGISTRY, Learner, make, planned  # noqa: F401
from . import classical, kernels, trees, nets, basis, meta, survey, rational, latent  # noqa: F401

# Declared, not yet implemented (each will move to a module above) -------------------------------------------
planned("smolyak", year=1963, field="numerical analysis (USSR)", ref="smolyak1963quadrature",
        note="design + interpolant; lives in upscale/ and samplers/")
planned("tt_cross", year=2010, field="numerical linear algebra", ref="oseledets2010tt", note="chooses its own samples")
planned("sobolev_mlp", year=2017, field="machine learning", ref="czarnecki2017sobolev", note="needs gradient data")
