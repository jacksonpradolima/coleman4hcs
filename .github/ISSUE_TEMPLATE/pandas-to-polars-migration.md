---
name: Pandas to Polars Migration
about: Complete migration plan from Pandas to Polars
title: '[MIGRATION] Migrate from Pandas to Polars'
labels: enhancement, migration, performance
assignees: ''
---

# Migration Plan: Pandas to Polars

## Overview

This issue tracks the migration of the Coleman4HCS codebase from **Pandas** to **Polars**. Polars is a modern DataFrame library that offers significant performance improvements (5-10x faster) through lazy evaluation, parallel processing, and efficient memory usage.

## Benefits

- **5-10x performance improvement** for DataFrame operations
- **Lower memory usage** through Apache Arrow format
- **Better code clarity** with explicit expression API
- **Parallel execution** out of the box
- **Future-proof codebase** with actively developed library

## Scope

### Files to Migrate (14 total)

#### Core Application Files (7 files)
- [ ] `coleman4hcs/utils/monitor.py` - Experiment data collection
- [ ] `coleman4hcs/scenarios.py` - CSV data loading and scenario management  
- [ ] `coleman4hcs/agent.py` - Agent action tracking and history
- [ ] `coleman4hcs/bandit.py` - Arms management in bandit models
- [ ] `coleman4hcs/policy.py` - Policy history tracking
- [ ] `main.py` - Main entry point, CSV operations
- [ ] `coleman4hcs/statistics/vargha_delaney.py` - Statistical analysis

#### Test Files (7 files)
- [ ] `tests/test_agent.py`
- [ ] `tests/test_bandit.py`
- [ ] `tests/test_environment.py`
- [ ] `tests/test_policy.py`
- [ ] `tests/test_scenarios.py`
- [ ] `tests/utils/test_monitor.py`
- [ ] `tests/statistics/test_varga_delaney.py`

## Migration Strategy

### Phase 1: Preparation (Week 1)
- [ ] Set up development environment with polars
- [ ] Update `requirements.txt` to replace `pandas==2.2.3` with `polars==1.13.1`
- [ ] Create helper functions for common patterns
- [ ] Run baseline tests and benchmarks

### Phase 2: Core Components (Weeks 2-3)
- [ ] Migrate `coleman4hcs/utils/monitor.py`
  - Replace `pd.DataFrame(columns=...)` with schema-based creation
  - Update `pd.concat()` to `pl.concat(..., how="vertical")`
  - Replace `.empty` with `.is_empty()` or `.height == 0`
  
- [ ] Migrate `coleman4hcs/scenarios.py`
  - Update CSV reading: `pd.read_csv(sep=';')` → `pl.read_csv(separator=';')`
  - Handle date parsing (dayfirst parameter needs custom solution)
  - Replace filtering: `df[df['col'] == val]` → `df.filter(pl.col('col') == val)`
  - Update `.to_dict('records')` to `.to_dicts()`

- [ ] Migrate `coleman4hcs/agent.py`
  - Define explicit schemas for DataFrames
  - Replace row appending pattern: `df.loc[len(df)] = [...]` with concat
  - Update `.values` to `.to_list()` or `.to_numpy()`
  - Replace `.fillna()` with `.fill_null()`
  - Update `.map()` and `.apply()` operations with Polars expressions

- [ ] Migrate `coleman4hcs/bandit.py`
  - Similar patterns to agent.py
  - Define schema for arms DataFrame

- [ ] Migrate `coleman4hcs/policy.py`
  - Handle history DataFrame similar to agent
  - May need `pl.List()` or `pl.Object` for nested features

### Phase 3: Application Layer (Week 4)
- [ ] Migrate `main.py`
  - Update `merge_csv()` function
  - Replace `.to_csv()` with `.write_csv()`
  - Verify DuckDB integration still works

- [ ] Migrate `coleman4hcs/statistics/vargha_delaney.py`
  - Replace `pd.Categorical()` with `.cast(pl.Categorical)`
  - Update `.copy()` to `.clone()`
  - Replace `sort_values(inplace=True)` with `.sort()`

### Phase 4: Testing & Validation (Week 5)
- [ ] Update all test fixtures
- [ ] Update test assertions to use `.frame_equal()`
- [ ] Run full test suite
- [ ] Performance benchmarking
- [ ] Fix edge cases
- [ ] Integration testing

### Phase 5: Documentation (Week 6)
- [ ] Update docstrings with Polars types
- [ ] Update README if needed
- [ ] Code review
- [ ] Final validation

## Key API Changes

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Import | `import pandas as pd` | `import polars as pl` |
| Create empty DF | `pd.DataFrame(columns=['a'])` | `pl.DataFrame(schema={'a': pl.Utf8})` |
| Read CSV | `pd.read_csv(file, sep=';')` | `pl.read_csv(file, separator=';')` |
| Write CSV | `df.to_csv(file, index=False)` | `df.write_csv(file)` |
| Filter | `df[df['col'] == val]` | `df.filter(pl.col('col') == val)` |
| Concat | `pd.concat([...], ignore_index=True)` | `pl.concat([...], how="vertical")` |
| Fill null | `df['col'].fillna(0)` | `df['col'].fill_null(0)` |
| Copy | `df.copy()` | `df.clone()` |
| Empty check | `df.empty` | `df.is_empty()` |
| To list | `df['col'].values` | `df['col'].to_list()` |
| To dict | `df.to_dict('records')` | `df.to_dicts()` |
| Unique count | `df['col'].nunique()` | `df['col'].n_unique()` |

## Detailed Documentation

See the following files for comprehensive migration guidance:

1. **`MIGRATION_PLAN_PANDAS_TO_POLARS.md`** - Complete migration strategy with:
   - Current state analysis
   - Benefits and risks
   - Detailed component-by-component migration guide
   - API mapping reference
   - Testing strategy
   - Rollout plan

2. **`MIGRATION_EXAMPLES.md`** - Detailed code examples including:
   - Advanced operations (row-wise updates, complex filtering)
   - String operations  
   - Merge and join operations
   - Null handling
   - Performance tips and best practices
   - Common gotchas and how to avoid them

3. **`POLARS_QUICK_REFERENCE.md`** - Quick cheat sheet for:
   - Side-by-side API comparison
   - Common patterns in Coleman4HCS
   - Type system overview
   - Testing guidelines
   - Migration checklist

## Risk Assessment

### High Risk Areas
1. **Date Parsing with `dayfirst=True`** - Needs custom solution or data preprocessing
2. **In-place Operations** - Must update all code to use returned DataFrames
3. **Row Appending Pattern** - Common `df.loc[len(df)] = [...]` pattern needs rewrite

### Mitigation
- Comprehensive testing at each phase
- Incremental migration with validation
- Keep pandas installed during transition for comparison testing
- Performance benchmarks to validate improvements

## Dependencies Update

**requirements.txt changes:**
```diff
 duckdb==1.1.3
 flake8==7.1.1
 numpy==2.0.1
-pandas==2.2.3
+polars==1.13.1
 pyarrow==18.1.0
```

**Note:** Keep pyarrow as Polars uses it internally.

## Success Criteria

- [ ] All tests pass
- [ ] Performance benchmarks show improvement (target: 3-5x faster)
- [ ] CSV files remain compatible
- [ ] DuckDB integration works
- [ ] No functional regressions
- [ ] Code quality maintained or improved

## Timeline

**Total Duration:** 6 weeks

- Week 1: Preparation and setup
- Weeks 2-3: Core components migration
- Week 4: Application layer migration  
- Week 5: Testing and validation
- Week 6: Documentation and deployment

## Additional Resources

- **Polars Documentation:** https://pola-rs.github.io/polars/
- **Polars User Guide:** https://pola-rs.github.io/polars-book/
- **Pandas to Polars Migration:** https://pola-rs.github.io/polars/user-guide/migration/pandas/
- **Performance Benchmarks:** https://www.pola.rs/benchmarks.html

## Questions or Concerns?

Please comment on this issue with any questions, concerns, or suggestions about the migration plan.

---

**Note:** This migration is expected to significantly improve performance while maintaining all existing functionality. The detailed migration documents provide comprehensive guidance for each step of the process.
