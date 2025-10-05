# Migration Plan: Pandas to Polars

## Executive Summary

This document provides a comprehensive migration plan for transitioning the Coleman4HCS codebase from **Pandas** to **Polars**. Polars is a modern DataFrame library that offers significant performance improvements through lazy evaluation, parallel processing, and efficient memory usage.

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Benefits of Migration](#benefits-of-migration)
3. [Migration Strategy](#migration-strategy)
4. [Detailed Component Migration](#detailed-component-migration)
5. [API Mapping Reference](#api-mapping-reference)
6. [Testing Strategy](#testing-strategy)
7. [Rollout Plan](#rollout-plan)
8. [Risk Assessment](#risk-assessment)

---

## Current State Analysis

### Files Using Pandas

The following files currently import and use pandas:

#### Core Application Files
1. **main.py** - Main entry point, CSV merging, data loading
2. **coleman4hcs/agent.py** - Agent action tracking and history management
3. **coleman4hcs/bandit.py** - Arms management in bandit models
4. **coleman4hcs/policy.py** - Policy history tracking
5. **coleman4hcs/scenarios.py** - CSV data loading and scenario management
6. **coleman4hcs/utils/monitor.py** - Experiment data collection and monitoring
7. **coleman4hcs/statistics/vargha_delaney.py** - Statistical analysis

#### Test Files
8. **tests/test_agent.py**
9. **tests/test_bandit.py**
10. **tests/test_environment.py**
11. **tests/test_policy.py**
12. **tests/test_scenarios.py**
13. **tests/statistics/test_varga_delaney.py**
14. **tests/utils/test_monitor.py**

### Pandas Operations Used

Based on code analysis, the following pandas operations are currently used:

1. **DataFrame Creation**
   - `pd.DataFrame(columns=[...])` - Creating empty DataFrames with column names
   - `pd.DataFrame(data, columns=[...])` - Creating DataFrames from lists/dicts
   - `.infer_objects()` - Type inference

2. **Data Reading/Writing**
   - `pd.read_csv(file, sep=';', parse_dates=[...], dayfirst=True)` - CSV reading
   - `.to_csv(file, sep=';', index=False)` - CSV writing
   - `.to_dict('records')` - Converting to list of dictionaries

3. **Data Manipulation**
   - `pd.concat([df1, df2], ignore_index=True)` - Concatenating DataFrames
   - `pd.to_numeric(column, errors="coerce").fillna(0)` - Type conversion
   - DataFrame filtering: `df[df['column'] == value]`
   - Column selection: `df[['col1', 'col2']]`
   - `.sum()` - Aggregation
   - `.unique()` - Getting unique values
   - `.copy()` - Copying DataFrames
   - `.empty` - Checking if DataFrame is empty
   - `.isna().all(axis=None)` - Null checking
   - `.map()` - Mapping values
   - `.apply()` - Applying functions
   - `.loc[]` - Location-based indexing
   - `.values` - Getting numpy array

4. **Special Operations**
   - `pd.Categorical()` - Categorical data handling
   - `pd.set_option('mode.chained_assignment', None)` - Configuration

---

## Benefits of Migration

### Performance Improvements
- **5-10x faster** for typical DataFrame operations
- **Parallel execution** out of the box
- **Lazy evaluation** for optimized query plans
- **Lower memory footprint** through Apache Arrow format

### Code Quality
- **More explicit API** reduces ambiguity
- **Better type safety** and schema enforcement
- **Consistent null handling** with `null` instead of `NaN`
- **Expression API** for cleaner transformations

### Maintenance
- **Active development** and modern architecture
- **Better documentation** for common operations
- **Rust-based core** for reliability and performance

---

## Migration Strategy

### Phase 1: Preparation (Week 1)
1. Set up development environment with polars
2. Create compatibility layer/wrapper if needed
3. Update requirements.txt
4. Create migration guide for developers

### Phase 2: Core Components (Weeks 2-3)
1. Migrate utility modules first (monitor.py)
2. Migrate data loading (scenarios.py)
3. Migrate agent and bandit modules
4. Migrate policy modules

### Phase 3: Main Application (Week 4)
1. Migrate main.py
2. Migrate statistics modules
3. Update any remaining helper utilities

### Phase 4: Testing & Validation (Week 5)
1. Update all test files
2. Run comprehensive test suite
3. Performance benchmarking
4. Fix any edge cases

### Phase 5: Documentation (Week 6)
1. Update all docstrings
2. Update README if applicable
3. Create migration notes
4. Update examples

---

## Detailed Component Migration

### 1. coleman4hcs/utils/monitor.py

**Current Pandas Usage:**
```python
import pandas as pd

class MonitorCollector:
    def __init__(self):
        self.col_names = ['scenario', 'experiment', 'step', ...]
        self.df = pd.DataFrame(columns=self.col_names)
        self.temp_rows = []
    
    def collect_from_temp(self):
        if self.temp_rows:
            batch_df = pd.DataFrame(self.temp_rows, columns=self.col_names)
            if not batch_df.empty and not batch_df.isna().all(axis=None):
                if self.df.empty:
                    self.df = batch_df
                else:
                    self.df = pd.concat([self.df, batch_df], ignore_index=True)
            self.temp_rows = []
```

**Polars Migration:**
```python
import polars as pl

class MonitorCollector:
    def __init__(self):
        self.col_names = ['scenario', 'experiment', 'step', ...]
        # Create empty DataFrame with schema
        schema = {col: pl.Utf8 for col in self.col_names}
        # Adjust schema based on actual data types:
        schema.update({
            'experiment': pl.Int64,
            'step': pl.Int64,
            'sched_time': pl.Float64,
            'detected': pl.Int64,
            # ... define proper types for all columns
        })
        self.df = pl.DataFrame(schema=schema)
        self.temp_rows = []
    
    def collect_from_temp(self):
        if self.temp_rows:
            batch_df = pl.DataFrame(self.temp_rows, schema=self.df.schema)
            # Polars is more explicit about null handling
            if batch_df.height > 0 and not batch_df.null_count().sum_horizontal()[0] == batch_df.height * batch_df.width:
                if self.df.height == 0:
                    self.df = batch_df
                else:
                    self.df = pl.concat([self.df, batch_df], how="vertical")
            self.temp_rows = []
```

**Key Changes:**
- Replace `pd.DataFrame(columns=...)` with `pl.DataFrame(schema=...)` with explicit types
- Replace `.empty` with `.height == 0` or `.is_empty()`
- Replace `pd.concat([...], ignore_index=True)` with `pl.concat([...], how="vertical")`
- Replace `.isna().all(axis=None)` with `.null_count()` operations

---

### 2. coleman4hcs/scenarios.py

**Current Pandas Usage:**
```python
def _read_testcases(self, tcfile: str) -> pd.DataFrame:
    df = pd.read_csv(tcfile, sep=';', parse_dates=['LastRun'], dayfirst=True)
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(0)
    return df

def get(self) -> Optional[VirtualScenario]:
    build_df = self.tcdf[self.tcdf["BuildId"] == self.current_build]
    testcases = build_df[self.REQUIRED_COLUMNS].to_dict('records')
    self.total_build_duration = build_df['Duration'].sum()
    # ...
```

**Polars Migration:**
```python
def _read_testcases(self, tcfile: str) -> pl.DataFrame:
    df = pl.read_csv(
        tcfile, 
        separator=';',
        try_parse_dates=True
    )
    # Polars has more explicit error handling
    df = df.with_columns([
        pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0)
    ])
    return df

def get(self) -> Optional[VirtualScenario]:
    build_df = self.tcdf.filter(pl.col("BuildId") == self.current_build)
    testcases = build_df.select(self.REQUIRED_COLUMNS).to_dicts()
    self.total_build_duration = build_df['Duration'].sum()
    # ...
```

**Key Changes:**
- Replace `pd.read_csv(sep=';', parse_dates=...)` with `pl.read_csv(separator=';', try_parse_dates=True)`
- Replace `pd.to_numeric(...).fillna()` with `.cast(...).fill_null()`
- Replace `df[df['col'] == value]` with `df.filter(pl.col('col') == value)`
- Replace `.to_dict('records')` with `.to_dicts()`
- Note: `dayfirst=True` parameter needs special handling - may need custom date parsing

---

### 3. coleman4hcs/agent.py

**Current Pandas Usage:**
```python
class Agent:
    def __init__(self, policy, bandit=None):
        self.col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'Q']
        self.actions = pd.DataFrame(columns=self.col_names).infer_objects()
    
    def add_action(self, action):
        if action not in self.actions['Name'].values:
            self.actions.loc[len(self.actions)] = [action, 0, 0, 0]
    
    def update_actions(self, actions):
        new_actions_df = pd.DataFrame({
            'Name': new_actions,
            'ActionAttempts': [0] * len(new_actions),
            'ValueEstimates': [0] * len(new_actions),
            'Q': [0] * len(new_actions)
        })
        self.actions = pd.concat([self.actions, new_actions_df], ignore_index=True)
```

**Polars Migration:**
```python
class Agent:
    def __init__(self, policy, bandit=None):
        self.col_names = ['Name', 'ActionAttempts', 'ValueEstimates', 'Q']
        schema = {
            'Name': pl.Utf8,
            'ActionAttempts': pl.Float64,
            'ValueEstimates': pl.Float64,
            'Q': pl.Float64
        }
        self.actions = pl.DataFrame(schema=schema)
    
    def add_action(self, action):
        if action not in self.actions['Name'].to_list():
            new_row = pl.DataFrame({
                'Name': [action],
                'ActionAttempts': [0.0],
                'ValueEstimates': [0.0],
                'Q': [0.0]
            })
            self.actions = pl.concat([self.actions, new_row], how="vertical")
    
    def update_actions(self, actions):
        new_actions_df = pl.DataFrame({
            'Name': new_actions,
            'ActionAttempts': [0.0] * len(new_actions),
            'ValueEstimates': [0.0] * len(new_actions),
            'Q': [0.0] * len(new_actions)
        })
        self.actions = pl.concat([self.actions, new_actions_df], how="vertical")
```

**Key Changes:**
- Replace `.infer_objects()` with explicit schema definition
- Replace `.values` with `.to_list()` or `.to_numpy()`
- Replace `.loc[len(df)] = [...]` with creating new DataFrame and concatenating
- Ensure numeric literals are floats (0.0) to match schema

---

### 4. coleman4hcs/bandit.py

**Current Pandas Usage:**
```python
class Bandit:
    def __init__(self, arms: List[Dict]):
        self.tc_fieldnames = ['Name', 'Duration', 'CalcPrio', ...]
        self.arms = pd.DataFrame(columns=self.tc_fieldnames)
        
    def update_arms(self, arms):
        new_arms = pd.DataFrame(arms, columns=self.tc_fieldnames)
        self.arms = pd.concat([self.arms, new_arms], ignore_index=True)
```

**Polars Migration:**
```python
class Bandit:
    def __init__(self, arms: List[Dict]):
        self.tc_fieldnames = ['Name', 'Duration', 'CalcPrio', ...]
        schema = {
            'Name': pl.Utf8,
            'Duration': pl.Float64,
            'CalcPrio': pl.Int64,
            # ... define all field types
        }
        self.arms = pl.DataFrame(schema=schema)
        
    def update_arms(self, arms):
        new_arms = pl.DataFrame(arms, schema=self.arms.schema)
        self.arms = pl.concat([self.arms, new_arms], how="vertical")
```

**Key Changes:**
- Define explicit schema instead of column names only
- Use `schema` parameter in DataFrame constructor
- Use `how="vertical"` in concat

---

### 5. coleman4hcs/policy.py

**Current Pandas Usage:**
```python
class LinUCBPolicy(Policy):
    def __init__(self, agent: ContextualAgent):
        self.hist_col_names = ['Name', 'Features', 'Reward', 'T']
        self.history = pd.DataFrame(columns=self.hist_col_names)
    
    def update_history(self, name, features, reward):
        new_entries = pd.DataFrame([...])
        self.history = pd.concat([self.history, new_entries], ignore_index=True)
```

**Polars Migration:**
```python
class LinUCBPolicy(Policy):
    def __init__(self, agent: ContextualAgent):
        self.hist_col_names = ['Name', 'Features', 'Reward', 'T']
        schema = {
            'Name': pl.Utf8,
            'Features': pl.List(pl.Float64),  # or pl.Object if needed
            'Reward': pl.Float64,
            'T': pl.Int64
        }
        self.history = pl.DataFrame(schema=schema)
    
    def update_history(self, name, features, reward):
        new_entries = pl.DataFrame([...], schema=self.history.schema)
        self.history = pl.concat([self.history, new_entries], how="vertical")
```

**Key Changes:**
- Handle nested data types with `pl.List()` or `pl.Object`
- Maintain schema consistency across operations

---

### 6. main.py

**Current Pandas Usage:**
```python
def merge_csv(files, output_file):
    df = pd.concat([pd.read_csv(file, sep=';') for file in files], ignore_index=True)
    df.to_csv(output_file, index=False, sep=';', quoting=csv.QUOTE_NONE)
    for file in files:
        os.remove(file)
```

**Polars Migration:**
```python
def merge_csv(files, output_file):
    dfs = [pl.read_csv(file, separator=';') for file in files]
    df = pl.concat(dfs, how="vertical")
    df.write_csv(output_file, separator=';')
    for file in files:
        os.remove(file)
```

**Key Changes:**
- Replace `sep` with `separator`
- Replace `.to_csv()` with `.write_csv()`
- Note: Polars doesn't have `index` parameter (no index concept)
- `quoting` parameter may need manual handling if required

---

### 7. coleman4hcs/statistics/vargha_delaney.py

**Current Pandas Usage:**
```python
pd.set_option('mode.chained_assignment', None)

def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    x = data.copy()
    if sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)
    
    groups = x[group_col].unique()
    
    return pd.DataFrame({
        'base': np.unique(data[group_col])[g1],
        'compared_with': np.unique(data[group_col])[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })
```

**Polars Migration:**
```python
# No need for chained assignment warning in Polars

def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    x = data.clone()
    if sort:
        # Polars handles categorical sorting differently
        x = x.with_columns([
            pl.col(group_col).cast(pl.Categorical)
        ]).sort([group_col, val_col])
    
    groups = x[group_col].unique()
    
    return pl.DataFrame({
        'base': data[group_col].unique().to_numpy()[g1],
        'compared_with': data[group_col].unique().to_numpy()[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })
```

**Key Changes:**
- Replace `.copy()` with `.clone()`
- Replace `sort_values(inplace=True)` with `.sort()` (returns new DataFrame)
- Use `.with_columns()` for transformations
- Replace `Categorical()` with `.cast(pl.Categorical)`
- Replace `.unique()` with `.unique().to_numpy()` when indexing needed

---

## API Mapping Reference

### DataFrame Creation

| Pandas | Polars |
|--------|--------|
| `pd.DataFrame(columns=['a', 'b'])` | `pl.DataFrame(schema={'a': pl.Utf8, 'b': pl.Int64})` |
| `pd.DataFrame(data, columns=['a'])` | `pl.DataFrame(data)` or `pl.DataFrame(data, schema=...)` |
| `.infer_objects()` | Not needed (automatic type inference) |

### Reading/Writing Data

| Pandas | Polars |
|--------|--------|
| `pd.read_csv(file, sep=';')` | `pl.read_csv(file, separator=';')` |
| `pd.read_csv(file, parse_dates=['col'])` | `pl.read_csv(file, try_parse_dates=True)` |
| `df.to_csv(file, index=False)` | `df.write_csv(file)` |
| `df.to_dict('records')` | `df.to_dicts()` |

### Data Manipulation

| Pandas | Polars |
|--------|--------|
| `df[df['col'] == value]` | `df.filter(pl.col('col') == value)` |
| `df[['col1', 'col2']]` | `df.select(['col1', 'col2'])` or `df[['col1', 'col2']]` |
| `pd.concat([df1, df2], ignore_index=True)` | `pl.concat([df1, df2], how="vertical")` |
| `df['col'].sum()` | `df['col'].sum()` or `df.select(pl.col('col').sum())` |
| `df['col'].unique()` | `df['col'].unique()` |
| `df.copy()` | `df.clone()` |
| `df.empty` | `df.is_empty()` or `df.height == 0` |
| `df['col'].values` | `df['col'].to_numpy()` or `df['col'].to_list()` |
| `df.loc[i] = [...]` | Create new row DataFrame and concat |
| `df['col'].fillna(0)` | `df['col'].fill_null(0)` |

### Type Operations

| Pandas | Polars |
|--------|--------|
| `pd.to_numeric(s, errors='coerce')` | `s.cast(pl.Float64, strict=False)` |
| `pd.Categorical(...)` | `.cast(pl.Categorical)` |
| `df['col'].map(dict)` | `df['col'].replace(dict)` or `.map_dict()` |
| `df['col'].apply(func)` | `df['col'].map_elements(func)` |

### Properties

| Pandas | Polars |
|--------|--------|
| `len(df)` or `df.shape[0]` | `df.height` or `len(df)` |
| `df.shape[1]` | `df.width` |
| `df.columns` | `df.columns` |
| `df.isna()` | `df.is_null()` |

---

## Testing Strategy

### 1. Unit Tests Update

For each module migration:
1. Update test fixtures to use Polars
2. Verify DataFrame equality with `assert df1.frame_equal(df2)`
3. Test edge cases (empty DataFrames, null values, etc.)

### 2. Integration Tests

1. End-to-end workflow testing
2. CSV reading/writing compatibility
3. Data type preservation

### 3. Performance Benchmarks

Create benchmarks to validate performance improvements:
```python
import pytest

@pytest.mark.benchmark(group="dataframes")
def test_benchmark_polars_concat(benchmark, large_dataframes):
    """Benchmark polars concat performance"""
    def concat_operation():
        return pl.concat(large_dataframes, how="vertical")
    
    result = benchmark(concat_operation)
    assert result.height > 0
```

### 4. Compatibility Testing

- Ensure DuckDB integration still works
- Verify CSV file formats remain compatible
- Check statistical analysis outputs

---

## Rollout Plan

### Dependencies Update

**requirements.txt changes:**
```diff
 duckdb==1.1.3
 flake8==7.1.1
 numpy==2.0.1
-pandas==2.2.3
+polars==1.13.1
 pyarrow==18.1.0
```

**Notes:**
- Keep pyarrow as Polars uses it internally
- NumPy integration works seamlessly with Polars

### Migration Order

1. **Week 1: Setup & Utils**
   - Update requirements.txt
   - Migrate coleman4hcs/utils/monitor.py
   - Update related tests

2. **Week 2: Data Layer**
   - Migrate coleman4hcs/scenarios.py
   - Update scenario tests
   - Verify CSV compatibility

3. **Week 3: Core Logic**
   - Migrate coleman4hcs/agent.py
   - Migrate coleman4hcs/bandit.py
   - Migrate coleman4hcs/policy.py
   - Update all related tests

4. **Week 4: Application Layer**
   - Migrate main.py
   - Migrate coleman4hcs/statistics/vargha_delaney.py
   - Integration testing

5. **Week 5: Testing & Validation**
   - Update all remaining test files
   - Performance benchmarking
   - Bug fixes and edge cases

6. **Week 6: Documentation & Cleanup**
   - Update docstrings
   - Update README
   - Code review and cleanup

---

## Risk Assessment

### High Risk Areas

1. **Date Parsing with `dayfirst=True`**
   - **Risk:** Polars may not support `dayfirst` parameter directly
   - **Mitigation:** Use custom date parsing or format specification

2. **In-place Operations**
   - **Risk:** Polars doesn't support in-place operations
   - **Mitigation:** Update code to use returned DataFrames

3. **Categorical Data**
   - **Risk:** Different categorical implementation
   - **Mitigation:** Test thoroughly, especially in vargha_delaney.py

4. **Row Appending Pattern**
   - **Risk:** `df.loc[len(df)] = [...]` pattern is common in codebase
   - **Mitigation:** Create helper function for row appending

### Medium Risk Areas

1. **DuckDB Integration**
   - **Risk:** DuckDB integration with Polars
   - **Mitigation:** DuckDB 1.1.3 supports Polars DataFrames well

2. **Test Data Fixtures**
   - **Risk:** All test fixtures need updating
   - **Mitigation:** Systematic update of all test files

3. **Performance Characteristics**
   - **Risk:** Different performance for small vs large DataFrames
   - **Mitigation:** Benchmark various data sizes

### Low Risk Areas

1. **Basic Operations**
   - Most common operations have direct equivalents
   - Well-documented migration path

2. **CSV I/O**
   - Both libraries handle CSV well
   - Minor parameter name changes only

---

## Migration Checklist

### Pre-Migration
- [ ] Create feature branch for migration
- [ ] Set up Polars in development environment
- [ ] Run full test suite to establish baseline
- [ ] Create performance benchmarks for comparison

### Core Migration
- [ ] Update requirements.txt
- [ ] Migrate coleman4hcs/utils/monitor.py
- [ ] Migrate coleman4hcs/scenarios.py
- [ ] Migrate coleman4hcs/agent.py
- [ ] Migrate coleman4hcs/bandit.py
- [ ] Migrate coleman4hcs/policy.py
- [ ] Migrate main.py
- [ ] Migrate coleman4hcs/statistics/vargha_delaney.py

### Test Migration
- [ ] Update tests/test_agent.py
- [ ] Update tests/test_bandit.py
- [ ] Update tests/test_environment.py
- [ ] Update tests/test_policy.py
- [ ] Update tests/test_scenarios.py
- [ ] Update tests/utils/test_monitor.py
- [ ] Update tests/statistics/test_varga_delaney.py

### Validation
- [ ] Run full test suite - all tests pass
- [ ] Performance benchmarks show improvement
- [ ] Integration tests pass
- [ ] CSV files remain compatible
- [ ] DuckDB integration works
- [ ] No regression in functionality

### Documentation
- [ ] Update docstrings with Polars types
- [ ] Update README if needed
- [ ] Create migration notes
- [ ] Update any code examples

### Deployment
- [ ] Code review
- [ ] Merge to main branch
- [ ] Update CI/CD if needed
- [ ] Monitor for issues

---

## Helper Functions

To ease migration, consider creating helper functions:

```python
# helpers/dataframe_utils.py
import polars as pl
from typing import List, Any, Dict

def append_row(df: pl.DataFrame, row: List[Any]) -> pl.DataFrame:
    """Helper to append a row to a DataFrame (mimics pandas .loc[len(df)] = row)"""
    new_row = pl.DataFrame([row], schema=df.schema)
    return pl.concat([df, new_row], how="vertical")

def append_rows(df: pl.DataFrame, rows: List[List[Any]]) -> pl.DataFrame:
    """Helper to append multiple rows to a DataFrame"""
    new_rows = pl.DataFrame(rows, schema=df.schema)
    return pl.concat([df, new_rows], how="vertical")

def is_empty(df: pl.DataFrame) -> bool:
    """Helper to check if DataFrame is empty (mimics pandas .empty)"""
    return df.height == 0

def to_records(df: pl.DataFrame) -> List[Dict]:
    """Helper to convert DataFrame to list of dicts (mimics pandas .to_dict('records'))"""
    return df.to_dicts()
```

---

## Additional Resources

1. **Polars Documentation:** https://pola-rs.github.io/polars/
2. **Polars User Guide:** https://pola-rs.github.io/polars-book/
3. **Pandas to Polars Guide:** https://pola-rs.github.io/polars/user-guide/migration/pandas/
4. **Performance Benchmarks:** https://www.pola.rs/benchmarks.html

---

## Conclusion

This migration from Pandas to Polars is expected to provide:
- **5-10x performance improvement** for DataFrame operations
- **Lower memory usage** through efficient data structures
- **Better code clarity** with explicit expression API
- **Future-proof codebase** with modern, actively developed library

The migration is straightforward with most operations having direct equivalents. The main challenges are:
1. Handling date parsing with `dayfirst=True`
2. Replacing in-place operations
3. Updating row append patterns

With careful execution following this plan, the migration can be completed in approximately 6 weeks with minimal disruption to ongoing development.
