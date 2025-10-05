# Pandas to Polars Migration - Implementation Summary

## 📋 Overview

This document provides a comprehensive summary of the migration planning documentation created for transitioning Coleman4HCS from Pandas to Polars.

## 📚 Documentation Structure

Four comprehensive documents have been created to guide the migration:

### 1. **MIGRATION_PLAN_PANDAS_TO_POLARS.md** (Main Strategy Document)
   - **Length:** ~760 lines
   - **Purpose:** Complete strategic planning document
   - **Contents:**
     - Current state analysis (14 files identified)
     - Detailed benefits and performance expectations
     - 6-week rollout plan with 5 phases
     - Component-by-component migration guides
     - Comprehensive API mapping reference
     - Testing strategy and validation approach
     - Risk assessment and mitigation strategies
     - Helper functions and utilities

### 2. **MIGRATION_EXAMPLES.md** (Code Examples)
   - **Length:** ~660 lines
   - **Purpose:** Practical code examples from actual codebase
   - **Contents:**
     - 10+ detailed migration examples
     - Advanced operations (row-wise updates, filtering)
     - String operations and regex handling
     - Merge and join operations
     - Null handling strategies
     - Performance optimization tips
     - Common gotchas and solutions
     - Testing considerations

### 3. **POLARS_QUICK_REFERENCE.md** (Cheat Sheet)
   - **Length:** ~350 lines
   - **Purpose:** Quick lookup reference
   - **Contents:**
     - Side-by-side API comparison
     - Common patterns in Coleman4HCS
     - Type system overview
     - Performance tips
     - Migration checklist
     - External resources

### 4. **MIGRATION_README.md** (Navigation Guide)
   - **Length:** ~250 lines
   - **Purpose:** Entry point and navigation
   - **Contents:**
     - Document overview and usage guide
     - Quick stats and timeline
     - Top 10 most common conversions
     - File migration priority order
     - Success criteria
     - Getting help resources

### 5. **.github/ISSUE_TEMPLATE/pandas-to-polars-migration.md** (GitHub Issue)
   - **Length:** ~280 lines
   - **Purpose:** Ready-to-use GitHub issue template
   - **Contents:**
     - Complete task checklist
     - Phase-by-phase breakdown
     - Success criteria
     - Documentation references

## 🎯 Key Findings

### Files Using Pandas

**Core Application Files (7):**
1. `coleman4hcs/utils/monitor.py` - DataFrame for experiment data collection
2. `coleman4hcs/scenarios.py` - CSV reading, data processing
3. `coleman4hcs/agent.py` - Agent actions tracking
4. `coleman4hcs/bandit.py` - Arms management
5. `coleman4hcs/policy.py` - Policy history
6. `main.py` - CSV operations, data merging
7. `coleman4hcs/statistics/vargha_delaney.py` - Statistical analysis

**Test Files (7):**
- `tests/test_agent.py`
- `tests/test_bandit.py`
- `tests/test_environment.py`
- `tests/test_policy.py`
- `tests/test_scenarios.py`
- `tests/utils/test_monitor.py`
- `tests/statistics/test_varga_delaney.py`

### Pandas Operations Identified

**Most Common Operations:**
1. `pd.DataFrame(columns=[...])` - Creating empty DataFrames (~8 instances)
2. `pd.read_csv(...)` - Reading CSV files (~5 instances)
3. `pd.concat([...], ignore_index=True)` - Concatenating DataFrames (~15 instances)
4. `.fillna()` / `.fill_null()` - Null handling (~8 instances)
5. DataFrame filtering with boolean indexing (~10 instances)
6. `.to_dict('records')` - Converting to dictionaries (~3 instances)
7. `.loc[]` assignment for row addition (~4 instances)
8. `.map()` and `.apply()` operations (~5 instances)
9. String operations with regex (~2 instances)
10. `.merge()` operations (~3 instances)

## 📊 Migration Impact Assessment

### Expected Benefits

**Performance:**
- 5-10x speed improvement for DataFrame operations
- Parallel execution without code changes
- Lazy evaluation for optimized query plans
- Lower memory footprint

**Code Quality:**
- More explicit API reduces ambiguity
- Better type safety with schema enforcement
- Clearer null handling semantics
- Modern, well-documented library

### Challenges Identified

**High Priority Issues:**
1. **Date Parsing:** `dayfirst=True` parameter has no direct equivalent
   - **Solution:** Custom date format specification or data preprocessing
   
2. **In-place Operations:** Polars doesn't support `inplace=True`
   - **Solution:** Always reassign returned DataFrames
   
3. **Row Appending:** Pattern `df.loc[len(df)] = [...]` needs rewriting
   - **Solution:** Create new DataFrame and concatenate

**Medium Priority Issues:**
1. **Test Fixtures:** All test data needs updating
2. **Type Strictness:** Polars requires explicit type handling
3. **API Differences:** Some operations have different names/patterns

## 🛠️ Migration Strategy

### Recommended Approach

**Phase 1: Preparation (Week 1)**
- Update dependencies
- Create helper functions
- Run baseline tests

**Phase 2: Core Components (Weeks 2-3)**
- Start with `utils/monitor.py` (simplest)
- Progress to `scenarios.py`
- Migrate `agent.py`, `bandit.py`, `policy.py`

**Phase 3: Application Layer (Week 4)**
- Migrate `main.py`
- Update statistics module

**Phase 4: Testing (Week 5)**
- Update all test files
- Run comprehensive validation
- Performance benchmarking

**Phase 5: Documentation (Week 6)**
- Update docstrings
- Final code review
- Deployment

### Critical Success Factors

1. **Incremental Migration:** One file at a time
2. **Continuous Testing:** Validate after each file
3. **Performance Validation:** Benchmark improvements
4. **Team Communication:** Regular progress updates

## 📝 API Mapping Summary

### Most Frequent Conversions

| Category | Pandas | Polars | Frequency |
|----------|--------|--------|-----------|
| Import | `import pandas as pd` | `import polars as pl` | 14 files |
| Create DF | `pd.DataFrame(columns=[...])` | `pl.DataFrame(schema={...})` | ~8 uses |
| Read CSV | `pd.read_csv(sep=';')` | `pl.read_csv(separator=';')` | ~5 uses |
| Filter | `df[df['col'] == val]` | `df.filter(pl.col('col') == val)` | ~10 uses |
| Concat | `pd.concat([...], ignore_index=True)` | `pl.concat([...], how="vertical")` | ~15 uses |
| Fill Null | `.fillna(value)` | `.fill_null(value)` | ~8 uses |
| Copy | `.copy()` | `.clone()` | ~5 uses |
| Empty Check | `.empty` | `.is_empty()` | ~3 uses |

## 🔍 Code Examples Highlights

### Example 1: DataFrame Creation
```python
# Before (Pandas)
self.df = pd.DataFrame(columns=self.col_names)

# After (Polars)
schema = {col: pl.Utf8 for col in self.col_names}
schema.update({'experiment': pl.Int64, 'step': pl.Int64, ...})
self.df = pl.DataFrame(schema=schema)
```

### Example 2: Row-wise Updates
```python
# Before (Pandas)
for test_case, r in zip(self.last_prioritization, reward):
    k = self.actions.loc[self.actions.Name == test_case, 'ActionAttempts'].values[0]
    self.actions.loc[self.actions.Name == test_case, 'ValueEstimates'] += alpha * (r - k)

# After (Polars)
reward_map = dict(zip(self.last_prioritization, reward))
self.actions = self.actions.with_columns([
    pl.when(pl.col('Name').is_in(list(reward_map.keys())))
    .then(pl.col('ValueEstimates') + (1.0 / pl.col('ActionAttempts')) * 
          (pl.col('Name').replace(reward_map, default=0.0) - pl.col('ValueEstimates')))
    .otherwise(pl.col('ValueEstimates'))
    .alias('ValueEstimates')
])
```

### Example 3: CSV Operations
```python
# Before (Pandas)
df = pd.read_csv(tcfile, sep=';', parse_dates=['LastRun'], dayfirst=True)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(0)

# After (Polars)
df = pl.read_csv(tcfile, separator=';')
df = df.with_columns([
    pl.col("LastRun").str.strptime(pl.Datetime, format="%d-%m-%Y %H:%M"),
    pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0.0)
])
```

## 📈 Expected Outcomes

### Performance Metrics
- **Target:** 3-5x faster operations (conservative estimate)
- **Best Case:** 5-10x improvement for certain operations
- **Memory:** 20-30% reduction in memory usage

### Quality Improvements
- **Type Safety:** Explicit schemas prevent type errors
- **Code Clarity:** Expression API more readable
- **Maintainability:** Modern library with active development

## 🚀 Next Steps

1. **Create GitHub Issue**
   - Use template from `.github/ISSUE_TEMPLATE/pandas-to-polars-migration.md`
   - Assign to team members
   - Set milestones for each phase

2. **Setup Development Environment**
   - Install Polars: `pip install polars==1.13.1`
   - Keep Pandas temporarily for comparison

3. **Begin Migration**
   - Start with `coleman4hcs/utils/monitor.py`
   - Follow phase-by-phase approach
   - Update tracking issue regularly

4. **Continuous Validation**
   - Run tests after each file migration
   - Compare outputs with Pandas version
   - Benchmark performance improvements

## 📚 Resource Links

### Official Polars Resources
- **Documentation:** https://pola-rs.github.io/polars/
- **User Guide:** https://pola-rs.github.io/polars-book/
- **API Reference:** https://pola-rs.github.io/polars/py-polars/html/reference/
- **Migration Guide:** https://pola-rs.github.io/polars/user-guide/migration/pandas/
- **GitHub:** https://github.com/pola-rs/polars

### Internal Documentation
- Main Plan: `MIGRATION_PLAN_PANDAS_TO_POLARS.md`
- Code Examples: `MIGRATION_EXAMPLES.md`
- Quick Reference: `POLARS_QUICK_REFERENCE.md`
- Navigation: `MIGRATION_README.md`
- Issue Template: `.github/ISSUE_TEMPLATE/pandas-to-polars-migration.md`

## ✅ Deliverables Checklist

- [x] Comprehensive migration plan document
- [x] Detailed code examples from actual codebase
- [x] Quick reference guide
- [x] Navigation/README document
- [x] GitHub issue template
- [x] Current state analysis (14 files identified)
- [x] API mapping reference
- [x] Risk assessment and mitigation strategies
- [x] Testing strategy
- [x] 6-week timeline with milestones
- [x] Helper functions and utilities
- [x] Performance optimization tips

## 📞 Support

For questions or issues during migration:

1. **Check Documentation:** Start with `MIGRATION_README.md`
2. **Find Examples:** Search `MIGRATION_EXAMPLES.md` for similar patterns
3. **Quick Lookup:** Use `POLARS_QUICK_REFERENCE.md` for API conversions
4. **Official Docs:** Consult Polars documentation for detailed API info

## 🎉 Conclusion

This comprehensive migration plan provides:
- **Complete Analysis:** All 14 files using Pandas identified
- **Detailed Strategy:** 6-week phased approach
- **Practical Examples:** Real code from Coleman4HCS
- **Quick Reference:** Easy API lookup
- **Risk Mitigation:** Identified challenges with solutions
- **Success Criteria:** Clear validation metrics

The migration is expected to deliver significant performance improvements while maintaining all existing functionality. With this detailed planning, the team has all the resources needed for a successful migration.

**Total Documentation:** ~2,500 lines across 5 documents
**Estimated Effort:** 6 weeks (1 person full-time or 2 people part-time)
**Expected ROI:** 5-10x performance improvement + better code quality
