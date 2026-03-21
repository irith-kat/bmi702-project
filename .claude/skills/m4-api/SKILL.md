---
name: m4-api
description: Use the M4 Python API to query clinical datasets programmatically. Use when writing code to access clinical databases, executing SQL via Python, or performing multi-step data analysis.
---

# M4 Python API

The M4 Python API provides programmatic access to clinical datasets, returning native Python types (DataFrames, dicts) suited for analysis pipelines.

## Required Workflow

**Always follow this sequence:**

1. `set_dataset()` — select the dataset (REQUIRED FIRST)
2. `get_schema()` / `get_table_info()` — explore available tables
3. `execute_query()` — run SQL queries

```python
from m4 import set_dataset, get_schema, get_table_info, execute_query

# Step 1: Always set dataset first
set_dataset("mimic-iv")  # or "mimic-iv-demo", "eicu", "mimic-iv-note"

# Step 2: Explore schema
schema = get_schema()
print(schema['tables'])  # List of table names

# Step 3: Inspect specific tables before querying
info = get_table_info("mimiciv_hosp.patients")
print(info['schema'])  # DataFrame with column names, types
print(info['sample'])  # DataFrame with sample rows

# Step 4: Execute queries — returns pd.DataFrame
df = execute_query("SELECT gender, COUNT(*) as n FROM mimiciv_hosp.patients GROUP BY gender")
```

## API Reference

### Dataset Management

| Function | Returns | Description |
|----------|---------|-------------|
| `list_datasets()` | `list[str]` | Available dataset names |
| `set_dataset(name)` | `str` | Set active dataset |
| `get_active_dataset()` | `str` | Get current dataset name |

### Tabular Data

| Function | Returns | Description |
|----------|---------|-------------|
| `get_schema()` | `dict` | `{'backend_info': str, 'tables': list[str]}` |
| `get_table_info(table, show_sample=True)` | `dict` | `{'schema': DataFrame, 'sample': DataFrame}` |
| `execute_query(sql)` | `DataFrame` | Query results as pandas DataFrame |

### Clinical Notes (requires `mimic-iv-note` dataset)

```python
set_dataset("mimic-iv-note")
```

| Function | Returns | Description |
|----------|---------|-------------|
| `search_notes(query, note_type, limit, snippet_length)` | `dict` | `{'results': dict[str, DataFrame]}` |
| `get_note(note_id, max_length)` | `dict` | `{'text': str, 'subject_id': int, ...}` |
| `list_patient_notes(subject_id, note_type, limit)` | `dict` | `{'notes': dict[str, DataFrame]}` |

## Error Handling

```python
from m4 import execute_query, set_dataset, DatasetError, QueryError, ModalityError

try:
    df = execute_query("SELECT * FROM mimiciv_hosp.patients")
except DatasetError:
    # No dataset selected, or dataset not found — call set_dataset() first
    set_dataset("mimic-iv")
    df = execute_query("SELECT * FROM mimiciv_hosp.patients")
except QueryError as e:
    # SQL error or table not found — check table name with get_schema()
    print(f"Query failed: {e}")
except ModalityError:
    # Notes function called without notes dataset — switch dataset
    set_dataset("mimic-iv-note")
```

## Dataset State

Dataset selection is module-level state that persists across calls. Call `set_dataset()` explicitly whenever switching between tabular and notes data.

```python
set_dataset("mimic-iv")
df1 = execute_query("SELECT COUNT(*) FROM mimiciv_hosp.patients")

set_dataset("mimic-iv-note")
df2 = execute_query("SELECT subject_id, text FROM mimiciv_note.discharge LIMIT 10")
```

## Notes on Table Names

All queries use canonical `schema.table` names (e.g., `mimiciv_hosp.patients`, `mimiciv_icu.icustays`). These work on both backends without modification.
