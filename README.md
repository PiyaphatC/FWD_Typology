## Overview

This repository contains the data processing pipeline and dataset used to develop a replicable, evidence-based structural typology for pavement maintenance prioritization. Using 77,230 Falling Weight Deflectometer (FWD) measurements across 440 Thai national highway sections over ten years (2014–2023), we apply K-means clustering on deflection bowl indices to classify pavement structural condition into three distinct types.

| Type | Label | Description |
|------|-------|-------------|
| **S** | Structurally Sound | Good structural condition, routine monitoring |
| **M** | Moderately Deteriorated | Early-stage intervention warranted |
| **C** | Critically Deteriorated | Near-complete base layer failure, urgent rehabilitation |

Key finding: **38.0% of the network** falls into Type C, with mean base modulus of 344 MPa, mean D0 of 661 μm, and mean remaining service life of 4.0 years. The proportion of critically deteriorated measurements remained at 20.7–42.0% annually from 2016–2023 with no downward trend.

---

## Repository Structure

```
fwd_typology_review/
├── cluster_pipeline.py       # K-means clustering pipeline on FWD deflection bowl indices
├── fwd100_latlon.csv         # FWD measurement dataset with geolocation (n = 77,230)
└── manuscript/
    └── Manoj_paper_rev11.docx  # Current manuscript draft
```

---

## Methodology

1. **Data:** FWD measurements from 440 Thai national highway sections (2014–2023)
2. **Features:** Four deflection bowl indices derived from FWD deflection basins
3. **Clustering:** K-means (k=3), validated against:
   - Back-calculated elastic moduli
   - Remaining service life (RSL)
   - Overlay thickness requirements
4. **Output:** Typology-to-intervention mapping for tiered budget allocation

---
