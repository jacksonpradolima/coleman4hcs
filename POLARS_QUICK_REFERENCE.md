# Pandas to Polars - Quick Reference Guide

A concise cheat sheet for the Coleman4HCS migration from Pandas to Polars.

## Installation

```bash
# Remove pandas, add polars
pip uninstall pandas
pip install polars==1.13.1
```

Update `requirements.txt`:
```diff
-pandas==2.2.3
+polars==1.13.1
```

---

## Import Statement

```python
# Before
import pandas as pd

# After  
import polars as pl
```

---

## Quick API Translation

### DataFrame Creation

```python
# Pandas
df = pd.DataFrame(columns=['Name', 'Value'])
df = pd.DataFrame(data, columns=['Name', 'Value'])

# Polars
df = pl.DataFrame(schema={'Name': pl.Utf8, 'Value': pl.Float64})
df = pl.DataFrame(data)
```

### CSV Operations

```python
# Pandas
df = pd.read_csv('file.csv', sep=';', parse_dates=['Date'])
df.to_csv('file.csv', index=False, sep=';')

# Polars
df = pl.read_csv('file.csv', separator=';', try_parse_dates=True)
df.write_csv('file.csv', separator=';')
```

### Filtering

```python
# Pandas
df[df['col'] == value]
df[df['col'] > 10]
df[(df['a'] > 5) & (df['b'] < 10)]

# Polars
df.filter(pl.col('col') == value)
df.filter(pl.col('col') > 10)
df.filter((pl.col('a') > 5) & (pl.col('b') < 10))
```

### Selecting Columns

```python
# Pandas
df[['col1', 'col2']]
df['col1']

# Polars
df.select(['col1', 'col2'])
df['col1']  # or df.select('col1')
```

### Adding/Modifying Columns

```python
# Pandas
df['new_col'] = df['col1'] + df['col2']
df['col'] = df['col'].fillna(0)

# Polars
df = df.with_columns([
    (pl.col('col1') + pl.col('col2')).alias('new_col')
])
df = df.with_columns([
    pl.col('col').fill_null(0)
])
```

### Concatenation

```python
# Pandas
pd.concat([df1, df2], ignore_index=True)

# Polars
pl.concat([df1, df2], how="vertical")
```

### Merging/Joining

```python
# Pandas
df1.merge(df2, on='key', how='left')

# Polars
df1.join(df2, on='key', how='left')
```

### Aggregations

```python
# Pandas
df['col'].sum()
df['col'].mean()
df['col'].max()
df.groupby('group').agg({'value': 'sum'})

# Polars
df['col'].sum()
df['col'].mean()
df['col'].max()
df.group_by('group').agg(pl.col('value').sum())
```

### Sorting

```python
# Pandas
df.sort_values('col', ascending=True)
df.sort_values(['col1', 'col2'], ascending=[True, False])

# Polars
df.sort('col')
df.sort(['col1', 'col2'], descending=[False, True])
```

### Unique Values

```python
# Pandas
df['col'].unique()
df['col'].nunique()

# Polars
df['col'].unique()
df['col'].n_unique()
```

### Null Handling

```python
# Pandas
df['col'].fillna(0)
df['col'].isna()
df.dropna()

# Polars
df.with_columns([pl.col('col').fill_null(0)])
df['col'].is_null()
df.drop_nulls()
```

### Type Conversion

```python
# Pandas
pd.to_numeric(df['col'], errors='coerce')
df['col'].astype(int)

# Polars
df['col'].cast(pl.Float64, strict=False)
df['col'].cast(pl.Int64)
```

### Copying

```python
# Pandas
df.copy()

# Polars
df.clone()
```

### Checking Empty

```python
# Pandas
df.empty
len(df) == 0

# Polars
df.is_empty()
df.height == 0
```

### Shape

```python
# Pandas
df.shape  # (rows, cols)
df.shape[0]  # rows
df.shape[1]  # cols

# Polars
(df.height, df.width)
df.height
df.width
```

### To List/Array

```python
# Pandas
df['col'].tolist()
df['col'].values

# Polars
df['col'].to_list()
df['col'].to_numpy()
```

### To Dict

```python
# Pandas
df.to_dict('records')

# Polars
df.to_dicts()
```

---

## String Operations

```python
# Pandas
df['col'].str.replace('old', 'new', regex=True)
df['col'].str.contains('pattern')
df['col'].str.lower()

# Polars
df['col'].str.replace_all('old', 'new')  # regex by default
df['col'].str.contains('pattern')
df['col'].str.to_lowercase()
```

---

## Common Patterns in Coleman4HCS

### Pattern 1: Appending Rows

```python
# Pandas
df.loc[len(df)] = [value1, value2, value3]

# Polars
new_row = pl.DataFrame({'col1': [value1], 'col2': [value2], 'col3': [value3]})
df = pl.concat([df, new_row], how="vertical")
```

### Pattern 2: Conditional Updates

```python
# Pandas
df.loc[df['Name'] == 'test', 'Value'] = 10

# Polars
df = df.with_columns([
    pl.when(pl.col('Name') == 'test')
      .then(10)
      .otherwise(pl.col('Value'))
      .alias('Value')
])
```

### Pattern 3: Map/Apply

```python
# Pandas
df['col'].map(mapping_dict)
df['col'].apply(lambda x: x * 2)

# Polars
df['col'].replace(mapping_dict)
df['col'].map_elements(lambda x: x * 2)  # Use with caution, slower
# Better: use expressions when possible
df.with_columns([(pl.col('col') * 2).alias('col')])
```

### Pattern 4: Filtering with isin

```python
# Pandas
df[~df['Name'].isin(obsolete_actions)]

# Polars
df.filter(~pl.col('Name').is_in(obsolete_actions))
```

---

## Type System

### Polars Data Types

```python
pl.Int8, pl.Int16, pl.Int32, pl.Int64
pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
pl.Float32, pl.Float64
pl.Utf8  # String
pl.Boolean
pl.Date
pl.Datetime
pl.Duration
pl.List(pl.Int64)  # List of integers
pl.Categorical
pl.Object  # Python objects (use sparingly)
```

---

## Key Differences

### 1. No Index
Polars doesn't have an index concept. Use filters or row numbers.

### 2. Immutability
All operations return new DataFrames. No `inplace=True`.

```python
# Pandas
df.sort_values('col', inplace=True)

# Polars
df = df.sort('col')  # Must reassign
```

### 3. Expression API
Polars uses expressions for transformations:

```python
# Build complex expressions
pl.col('value').filter(pl.col('value') > 0).sum()
```

### 4. Lazy Evaluation
Use `scan_csv` instead of `read_csv` for lazy loading:

```python
# Lazy
df = pl.scan_csv('file.csv').filter(...).select(...).collect()

# Eager
df = pl.read_csv('file.csv').filter(...).select(...)
```

---

## Performance Tips

1. **Use expressions instead of loops**
   ```python
   # Slow
   for i in range(len(df)):
       df = df.with_columns([...])
   
   # Fast
   df = df.with_columns([pl.col('x').expression_chain()])
   ```

2. **Batch operations**
   ```python
   # Create mappings/lookups once
   mapping = {k: v for k, v in ...}
   df = df.with_columns([pl.col('x').replace(mapping)])
   ```

3. **Use lazy evaluation for large datasets**
   ```python
   result = pl.scan_csv('large.csv').filter(...).collect()
   ```

4. **Define schemas explicitly**
   ```python
   schema = {'col1': pl.Int64, 'col2': pl.Utf8}
   df = pl.DataFrame(data, schema=schema)
   ```

---

## Testing

### DataFrame Equality

```python
# Pandas
pd.testing.assert_frame_equal(df1, df2)

# Polars
assert df1.frame_equal(df2)
```

### Series Equality

```python
# Pandas
pd.testing.assert_series_equal(s1, s2)

# Polars
assert s1.series_equal(s2)
```

---

## Common Gotchas

### 1. Float vs Int
Be explicit with numeric types:
```python
# Pandas auto-converts
df['int_col'] = 0

# Polars needs matching type
df = df.with_columns([pl.lit(0.0).alias('float_col')])
```

### 2. Null vs NaN
```python
# Pandas uses NaN
df['col'].fillna(np.nan)

# Polars uses null
df.with_columns([pl.col('col').fill_null(None)])
```

### 3. String methods
```python
# Pandas
df['col'].str.method()

# Polars (similar but different methods)
df['col'].str.method()  # Check documentation for exact names
```

---

## Migration Checklist

- [ ] Update imports: `import pandas as pd` → `import polars as pl`
- [ ] Replace `pd.DataFrame(columns=[...])` with schema definition
- [ ] Update `pd.read_csv(sep=';')` to `pl.read_csv(separator=';')`
- [ ] Change `.to_csv(index=False)` to `.write_csv()`
- [ ] Replace `df[df['col'] == val]` with `df.filter(pl.col('col') == val)`
- [ ] Update `pd.concat([...], ignore_index=True)` to `pl.concat([...], how="vertical")`
- [ ] Change `.fillna()` to `.fill_null()`
- [ ] Replace `.copy()` with `.clone()`
- [ ] Update `.empty` to `.is_empty()` or `.height == 0`
- [ ] Change `.values` to `.to_list()` or `.to_numpy()`
- [ ] Replace `.to_dict('records')` with `.to_dicts()`
- [ ] Update `.nunique()` to `.n_unique()`
- [ ] Change `.loc[len(df)] = [...]` to DataFrame concatenation
- [ ] Update test assertions to use `.frame_equal()`

---

## Resources

- **Official Docs:** https://pola-rs.github.io/polars/
- **User Guide:** https://pola-rs.github.io/polars-book/
- **API Reference:** https://pola-rs.github.io/polars/py-polars/html/reference/
- **Migration Guide:** https://pola-rs.github.io/polars/user-guide/migration/pandas/
- **GitHub:** https://github.com/pola-rs/polars

---

## Need Help?

For Coleman4HCS specific questions, refer to:
- `MIGRATION_PLAN_PANDAS_TO_POLARS.md` - Complete migration strategy
- `MIGRATION_EXAMPLES.md` - Detailed code examples from actual files
