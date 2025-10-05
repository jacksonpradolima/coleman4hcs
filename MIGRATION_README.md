# Pandas to Polars Migration Documentation

This directory contains comprehensive documentation for migrating the Coleman4HCS codebase from Pandas to Polars.

## Quick Start

If you're ready to start the migration, begin here:

1. **Read**: [`POLARS_QUICK_REFERENCE.md`](POLARS_QUICK_REFERENCE.md) - Quick cheat sheet
2. **Plan**: [`MIGRATION_PLAN_PANDAS_TO_POLARS.md`](MIGRATION_PLAN_PANDAS_TO_POLARS.md) - Overall strategy
3. **Code**: [`MIGRATION_EXAMPLES.md`](MIGRATION_EXAMPLES.md) - Detailed code examples
4. **Track**: [`.github/ISSUE_TEMPLATE/pandas-to-polars-migration.md`](.github/ISSUE_TEMPLATE/pandas-to-polars-migration.md) - GitHub issue template

## Document Overview

### 1. POLARS_QUICK_REFERENCE.md
**Best for:** Quick lookups during coding

**Contents:**
- Side-by-side API comparison (Pandas vs Polars)
- Common patterns specific to Coleman4HCS
- Type system overview
- Quick migration checklist
- Performance tips

**Use this when:** You need to quickly find the Polars equivalent of a Pandas operation.

---

### 2. MIGRATION_PLAN_PANDAS_TO_POLARS.md
**Best for:** Understanding the overall strategy and timeline

**Contents:**
- Complete current state analysis (14 files using Pandas)
- Benefits and motivation for migration
- Detailed 6-week rollout plan
- Component-by-component migration guide
- API mapping reference
- Testing strategy
- Risk assessment

**Use this when:** You need to understand the big picture, plan resources, or communicate with stakeholders.

---

### 3. MIGRATION_EXAMPLES.md
**Best for:** Solving specific migration challenges

**Contents:**
- 10+ detailed examples from actual codebase files
- Advanced operations (row-wise updates, complex filtering)
- String operations with regex
- Merge and join operations
- Null handling strategies
- Performance optimization tips
- Common gotchas and how to avoid them
- Testing considerations

**Use this when:** You're migrating a specific file and need concrete examples of how to handle complex operations.

---

### 4. .github/ISSUE_TEMPLATE/pandas-to-polars-migration.md
**Best for:** Tracking progress and team coordination

**Contents:**
- Pre-formatted GitHub issue template
- Checklist of all files to migrate
- Phase-by-phase task breakdown
- Success criteria
- Timeline and milestones

**Use this when:** You're ready to create a GitHub issue to track the migration work.

---

## Migration Quick Stats

### Files Affected
- **14 total files** using Pandas
  - 7 core application files
  - 7 test files

### Key Benefits
- **5-10x performance improvement** expected
- **Lower memory usage** through Apache Arrow
- **Better code clarity** with expression API
- **Parallel execution** out of the box

### Timeline
- **6 weeks total**
- Week 1: Preparation
- Weeks 2-3: Core components
- Week 4: Application layer
- Week 5: Testing & validation
- Week 6: Documentation

---

## Most Common Pandas → Polars Conversions

Here are the top 10 most frequent changes needed:

1. **Import**
   ```python
   # Before
   import pandas as pd
   
   # After
   import polars as pl
   ```

2. **DataFrame Creation**
   ```python
   # Before
   df = pd.DataFrame(columns=['Name', 'Value'])
   
   # After
   df = pl.DataFrame(schema={'Name': pl.Utf8, 'Value': pl.Float64})
   ```

3. **CSV Reading**
   ```python
   # Before
   df = pd.read_csv('file.csv', sep=';')
   
   # After
   df = pl.read_csv('file.csv', separator=';')
   ```

4. **Filtering**
   ```python
   # Before
   df[df['col'] == value]
   
   # After
   df.filter(pl.col('col') == value)
   ```

5. **Concatenation**
   ```python
   # Before
   pd.concat([df1, df2], ignore_index=True)
   
   # After
   pl.concat([df1, df2], how="vertical")
   ```

6. **Null Handling**
   ```python
   # Before
   df['col'].fillna(0)
   
   # After
   df.with_columns([pl.col('col').fill_null(0)])
   ```

7. **Copying**
   ```python
   # Before
   df.copy()
   
   # After
   df.clone()
   ```

8. **Empty Check**
   ```python
   # Before
   df.empty
   
   # After
   df.is_empty()
   ```

9. **To List**
   ```python
   # Before
   df['col'].values
   
   # After
   df['col'].to_list()
   ```

10. **To Dict**
    ```python
    # Before
    df.to_dict('records')
    
    # After
    df.to_dicts()
    ```

---

## Files to Migrate (Priority Order)

### Phase 1: Utilities (Easiest)
1. ✅ `coleman4hcs/utils/monitor.py` - Start here!

### Phase 2: Data Layer
2. ✅ `coleman4hcs/scenarios.py`

### Phase 3: Core Logic
3. ✅ `coleman4hcs/bandit.py`
4. ✅ `coleman4hcs/agent.py`
5. ✅ `coleman4hcs/policy.py`

### Phase 4: Application
6. ✅ `main.py`
7. ✅ `coleman4hcs/statistics/vargha_delaney.py`

### Phase 5: Tests
8. ✅ All test files

---

## Getting Help

### During Migration

**Stuck on a specific conversion?**
→ Check [`MIGRATION_EXAMPLES.md`](MIGRATION_EXAMPLES.md) for similar examples

**Need to understand a Polars API?**
→ See [`POLARS_QUICK_REFERENCE.md`](POLARS_QUICK_REFERENCE.md) for API comparison

**Planning the migration?**
→ Review [`MIGRATION_PLAN_PANDAS_TO_POLARS.md`](MIGRATION_PLAN_PANDAS_TO_POLARS.md)

### External Resources

- **Official Documentation:** https://pola-rs.github.io/polars/
- **User Guide:** https://pola-rs.github.io/polars-book/
- **Pandas Migration Guide:** https://pola-rs.github.io/polars/user-guide/migration/pandas/
- **API Reference:** https://pola-rs.github.io/polars/py-polars/html/reference/

---

## Success Criteria

Migration is complete when:

- [ ] All 14 files migrated
- [ ] All tests passing
- [ ] Performance benchmarks show 3-5x improvement
- [ ] CSV compatibility maintained
- [ ] DuckDB integration working
- [ ] No functional regressions
- [ ] Documentation updated

---

## Contributing to Migration

When migrating a file:

1. Create a feature branch
2. Migrate one file at a time
3. Update corresponding tests
4. Run tests to verify
5. Commit with descriptive message
6. Update the tracking issue

---

## Notes

- Keep `pyarrow` in requirements.txt (used by Polars internally)
- DuckDB 1.1.3 has excellent Polars support
- Consider lazy evaluation (`pl.scan_csv`) for large datasets
- Define schemas explicitly for best performance

---

**Last Updated:** 2024
**Status:** Planning Phase
**Estimated Completion:** 6 weeks from start
