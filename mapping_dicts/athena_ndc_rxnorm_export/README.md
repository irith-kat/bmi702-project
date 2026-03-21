# OMOP Athena NDC + RxNorm Export

Vocabulary export from [OMOP Athena](https://athena.ohdsi.org/), used as input to `build_ndc_to_rxnorm.py`.

## Updating

1. Go to https://athena.ohdsi.org/ and log in (free account required).
2. Download vocabularies: **NDC** and **RxNorm**.
3. Unzip into this directory, replacing the existing files.
4. Regenerate the lookup tables:
   ```bash
   uv run python mapping_dicts/build_ndc_to_rxnorm.py
   ```
