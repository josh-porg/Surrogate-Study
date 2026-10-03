"""Learner library. Importing registers every implemented learner in REGISTRY; methods named in the coverage
matrix but not yet implemented are listed in PLANNED with their provenance."""
from .base import PLANNED, REGISTRY, Learner, make, planned  # noqa: F401
from . import classical, kernels, trees, nets, basis, meta  # noqa: F401

# Declared, not yet implemented (each will move to a module above) -------------------------------------------
planned("natural_neighbor", year=1981, field="computational geometry", ref="sibson1981brief")
planned("smolyak", year=1963, field="numerical analysis (USSR)", ref="smolyak1963quadrature",
        note="design + interpolant; lives in upscale/ and samplers/")
planned("bart", year=2010, field="Bayesian statistics", ref="chipman2010bart")
planned("sparse_gp", year=2009, field="machine learning", ref="titsias2009variational")
planned("tt_cross", year=2010, field="numerical linear algebra", ref="oseledets2010tt", note="chooses its own samples")
planned("fsvd2d", year=2013, field="numerical analysis", ref="townsend2013extension")
planned("sindy_ae_static", year=2019, field="dynamical systems / ROM", ref="champion2019data")
planned("kernel_flow_gp", year=2019, field="applied mathematics", ref="owhadi2019kernel")
planned("warped_gp", year=2014, field="machine learning", ref="snoek2014input")
planned("sobolev_mlp", year=2017, field="machine learning", ref="czarnecki2017sobolev", note="needs gradient data")
planned("aaa_rational", year=2018, field="numerical analysis", ref="nakatsukasa2018aaa", note="1-D/MF problems only")
