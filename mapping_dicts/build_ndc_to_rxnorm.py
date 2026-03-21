"""
Build NDC → RxNorm ingredient lookup table from OMOP Athena export.

Two output files are written to the same directory as this script:
  - ndc_to_rxnorm_ingredient.csv        : 11-digit NDC → RxNorm ingredient
  - drug_name_to_rxnorm_ingredient.csv  : MIMIC prescriptions.drug (lowercase) → RxNorm ingredient
    (derived from gcpt_drug_ndc.csv; used as fallback in rollup_ndc_to_ingredient)

Mapping path for NDC (two hops):
  NDC concept  --[Maps to]--> RxNorm drug  --[ANCESTOR]--> RxNorm Ingredient
  The CONCEPT_ANCESTOR table provides the transitive drug→ingredient link because
  "RxNorm has ing" is sparse in typical Athena exports.

  For multi-ingredient drugs (e.g. combination pills), the ancestor table may return
  multiple ingredients; we pick the closest ancestor (min_levels_of_separation) and
  among ties keep the first alphabetically to ensure a deterministic single mapping.

Run:
  uv run python mapping_dicts/build_ndc_to_rxnorm.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ATHENA = HERE / "athena_ndc_rxnorm_export"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading CONCEPT.csv...")
    concept = pd.read_csv(ATHENA / "CONCEPT.csv", sep="\t", dtype=str, low_memory=False)

    print("Loading CONCEPT_RELATIONSHIP.csv...")
    rel = pd.read_csv(
        ATHENA / "CONCEPT_RELATIONSHIP.csv", sep="\t", dtype=str, low_memory=False
    )
    rel = rel[rel["invalid_reason"].isna()].copy()

    print("Loading CONCEPT_ANCESTOR.csv...")
    anc = pd.read_csv(
        ATHENA / "CONCEPT_ANCESTOR.csv", sep="\t", dtype=str, low_memory=False
    )

    return concept, rel, anc


def build_drug_to_ingredient(concept: pd.DataFrame, anc: pd.DataFrame) -> pd.DataFrame:
    """
    Map any RxNorm drug concept_id → closest ingredient ancestor.
    Returns DataFrame with columns: drug_id, ingredient_id, ingredient_name.
    """
    ingredients = concept[
        (concept["vocabulary_id"] == "RxNorm")
        & (concept["concept_class_id"] == "Ingredient")
    ][["concept_id", "concept_name"]].rename(
        columns={"concept_id": "ingredient_id", "concept_name": "ingredient_name"}
    )
    ing_ids = set(ingredients["ingredient_id"])

    # Filter ancestor table to rows where ancestor is an ingredient
    drug_anc = anc[anc["ancestor_concept_id"].isin(ing_ids)].copy()
    drug_anc["min_levels_of_separation"] = drug_anc["min_levels_of_separation"].astype(
        int
    )

    # Pick closest ingredient per drug (min sep), break ties alphabetically on name
    drug_anc = drug_anc.merge(
        ingredients, left_on="ancestor_concept_id", right_on="ingredient_id"
    )
    drug_anc = (
        drug_anc.sort_values(
            ["descendant_concept_id", "min_levels_of_separation", "ingredient_name"]
        )
        .drop_duplicates(subset="descendant_concept_id", keep="first")
        .rename(columns={"descendant_concept_id": "drug_id"})
    )

    # Also include ingredients mapping to themselves (sep=0)
    self_map = ingredients.rename(columns={"ingredient_id": "drug_id"}).copy()
    self_map["ingredient_id"] = self_map["drug_id"]

    combined = pd.concat(
        [drug_anc[["drug_id", "ingredient_id", "ingredient_name"]], self_map],
        ignore_index=True,
    ).drop_duplicates(subset="drug_id", keep="first")

    return combined


def build_ndc_mapping(
    concept: pd.DataFrame, rel: pd.DataFrame, drug_to_ing: pd.DataFrame
) -> pd.DataFrame:
    """11-digit NDC concept_code → ingredient via Maps to + ancestor."""
    ndc_concepts = concept[
        (concept["vocabulary_id"] == "NDC") & (concept["concept_code"].str.len() == 11)
    ][["concept_id", "concept_code"]].rename(
        columns={"concept_id": "ndc_concept_id", "concept_code": "ndc"}
    )

    maps_to = rel[rel["relationship_id"] == "Maps to"][
        ["concept_id_1", "concept_id_2"]
    ].rename(columns={"concept_id_1": "ndc_concept_id", "concept_id_2": "drug_id"})

    result = (
        ndc_concepts.merge(maps_to, on="ndc_concept_id")
        .merge(drug_to_ing, on="drug_id")
        .drop(columns=["ndc_concept_id", "drug_id"])
        .drop_duplicates(subset="ndc", keep="first")
    )
    return result[["ndc", "ingredient_id", "ingredient_name"]]


def build_drug_name_mapping(
    drug_to_ing: pd.DataFrame,
) -> pd.DataFrame:
    """
    MIMIC prescriptions.drug (free-text name) → ingredient via gcpt_drug_ndc.csv.

    gcpt_drug_ndc.csv maps hospital formulary strings (concept_code) to OMOP concept
    IDs. The keys are normalized to lowercase for case-insensitive matching at query
    time. Used as fallback when NDC lookup fails.
    """
    gcpt = pd.read_csv(HERE / "gcpt_drug_ndc.csv", dtype=str)
    gcpt = gcpt[gcpt["target_concept_id"].notna()][
        ["concept_code", "target_concept_id"]
    ].rename(columns={"concept_code": "drug_name_raw", "target_concept_id": "drug_id"})
    gcpt["drug_name"] = gcpt["drug_name_raw"].str.strip().str.lower()

    result = (
        gcpt.merge(drug_to_ing, on="drug_id")
        .drop(columns=["drug_id", "drug_name_raw"])
        .drop_duplicates(subset="drug_name", keep="first")
    )
    return result[["drug_name", "ingredient_id", "ingredient_name"]]


def main():
    concept, rel, anc = load_tables()

    print("Building RxNorm drug → ingredient map via CONCEPT_ANCESTOR...")
    drug_to_ing = build_drug_to_ingredient(concept, anc)
    print(f"  {len(drug_to_ing):,} drug → ingredient entries")

    print("Building NDC → ingredient map...")
    ndc_map = build_ndc_mapping(concept, rel, drug_to_ing)
    out_ndc = HERE / "ndc_to_rxnorm_ingredient.csv"
    ndc_map.to_csv(out_ndc, index=False)
    print(f"  {len(ndc_map):,} NDC entries → {out_ndc}")

    print("Building drug-name → ingredient map (gcpt fallback)...")
    drug_name_map = build_drug_name_mapping(drug_to_ing)
    out_drug_name = HERE / "drug_name_to_rxnorm_ingredient.csv"
    drug_name_map.to_csv(out_drug_name, index=False)
    print(f"  {len(drug_name_map):,} drug-name entries → {out_drug_name}")


if __name__ == "__main__":
    main()
