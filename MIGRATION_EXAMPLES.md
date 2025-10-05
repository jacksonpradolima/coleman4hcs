# Pandas to Polars Migration - Detailed Code Examples

This document provides detailed, file-specific migration examples for the Coleman4HCS codebase.

## Table of Contents

1. [Advanced Operations](#advanced-operations)
2. [Complex Filtering and Selection](#complex-filtering-and-selection)
3. [Row-wise Updates](#row-wise-updates)
4. [String Operations](#string-operations)
5. [Merge and Join Operations](#merge-and-join-operations)
6. [Null Handling](#null-handling)
7. [Performance Tips](#performance-tips)

---

## Advanced Operations

### Example 1: agent.py - Row-wise Value Updates

**Current Pandas Code:**
```python
def observe(self, reward):
    self.update_action_attempts()
    
    for test_case, r in zip(self.last_prioritization, reward):
        # Get current values
        k = self.actions.loc[self.actions.Name == test_case, 'ActionAttempts'].values[0]
        q = self.actions.loc[self.actions.Name == test_case, 'ValueEstimates'].values[0]
        
        alpha = 1. / k
        
        # Update Q value
        self.actions.loc[self.actions.Name == test_case, 'ValueEstimates'] += alpha * (r - q)
    
    self.t += 1
```

**Polars Migration Option 1 - Using with_columns:**
```python
def observe(self, reward):
    self.update_action_attempts()
    
    # Create a mapping of test cases to rewards
    reward_map = dict(zip(self.last_prioritization, reward))
    
    # Update all rows at once using expressions
    self.actions = self.actions.with_columns([
        pl.when(pl.col('Name').is_in(list(reward_map.keys())))
        .then(
            pl.col('ValueEstimates') + 
            (1.0 / pl.col('ActionAttempts')) * 
            (pl.col('Name').replace(reward_map, default=0.0) - pl.col('ValueEstimates'))
        )
        .otherwise(pl.col('ValueEstimates'))
        .alias('ValueEstimates')
    ])
    
    self.t += 1
```

**Polars Migration Option 2 - Using loop (simpler but slightly slower):**
```python
def observe(self, reward):
    self.update_action_attempts()
    
    for test_case, r in zip(self.last_prioritization, reward):
        # Filter to get the specific row
        mask = pl.col('Name') == test_case
        row_data = self.actions.filter(mask)
        
        if row_data.height > 0:
            k = row_data['ActionAttempts'][0]
            q = row_data['ValueEstimates'][0]
            alpha = 1.0 / k
            
            # Update the DataFrame
            self.actions = self.actions.with_columns([
                pl.when(mask)
                .then(q + alpha * (r - q))
                .otherwise(pl.col('ValueEstimates'))
                .alias('ValueEstimates')
            ])
    
    self.t += 1
```

---

### Example 2: agent.py - Filtering with isin

**Current Pandas Code:**
```python
def update_actions(self, actions):
    current_actions = set(self.actions['Name'].values)
    new_actions = set(actions) - current_actions
    obsolete_actions = current_actions - set(actions)
    
    # Remove obsolete actions
    if obsolete_actions:
        self.actions = self.actions[~self.actions['Name'].isin(obsolete_actions)]
    
    # Add new actions
    if new_actions:
        new_actions_df = pd.DataFrame({
            'Name': list(new_actions),
            'ActionAttempts': 0,
            'ValueEstimates': 0,
            'Q': 0
        })
        self.actions = pd.concat([self.actions, new_actions_df], ignore_index=True)
```

**Polars Migration:**
```python
def update_actions(self, actions):
    current_actions = set(self.actions['Name'].to_list())
    new_actions = set(actions) - current_actions
    obsolete_actions = current_actions - set(actions)
    
    # Remove obsolete actions
    if obsolete_actions:
        self.actions = self.actions.filter(
            ~pl.col('Name').is_in(list(obsolete_actions))
        )
    
    # Add new actions
    if new_actions:
        new_actions_df = pl.DataFrame({
            'Name': list(new_actions),
            'ActionAttempts': [0.0] * len(new_actions),
            'ValueEstimates': [0.0] * len(new_actions),
            'Q': [0.0] * len(new_actions)
        })
        self.actions = pl.concat([self.actions, new_actions_df], how="vertical")
```

---

### Example 3: agent.py - Map and Apply Operations

**Current Pandas Code:**
```python
def update_action_attempts(self):
    state_size = len(self.last_prioritization)
    weights = np.linspace(1.0, 1e-12, state_size)
    index_map = {name: idx for idx, name in enumerate(self.last_prioritization)}
    
    self.actions['ActionAttempts'] += self.actions['Name'].map(index_map).apply(
        lambda idx: weights[idx] if idx is not None else 0)
```

**Polars Migration:**
```python
def update_action_attempts(self):
    state_size = len(self.last_prioritization)
    weights = np.linspace(1.0, 1e-12, state_size)
    index_map = {name: idx for idx, name in enumerate(self.last_prioritization)}
    
    # Create weight mapping
    weight_map = {name: weights[idx] for name, idx in index_map.items()}
    
    # Update using replace and default value
    self.actions = self.actions.with_columns([
        (pl.col('ActionAttempts') + 
         pl.col('Name').replace(weight_map, default=0.0))
        .alias('ActionAttempts')
    ])
```

---

## Complex Filtering and Selection

### Example 4: scenarios.py - Multiple Filters and String Operations

**Current Pandas Code:**
```python
def _read_variants(self, variantsfile: str) -> pd.DataFrame:
    df = pd.read_csv(variantsfile, sep=';', parse_dates=['LastRun'])
    
    # Remove weird characters
    df["Variant"] = df["Variant"].str.replace(r'[!#$%^&*()[]{};:,.<>?|`~=+]', '_', regex=True)
    
    return df

def get(self):
    # ... 
    variants = self.variants[self.variants["BuildId"] == self.current_build]
    # ...
```

**Polars Migration:**
```python
def _read_variants(self, variantsfile: str) -> pl.DataFrame:
    df = pl.read_csv(variantsfile, separator=';', try_parse_dates=True)
    
    # Remove weird characters using replace_all
    df = df.with_columns([
        pl.col("Variant").str.replace_all(r'[!#$%^&*()[]{};:,.<>?|`~=+]', '_')
    ])
    
    return df

def get(self):
    # ... 
    variants = self.variants.filter(pl.col("BuildId") == self.current_build)
    # ...
```

---

### Example 5: scenarios.py - nunique and unique

**Current Pandas Code:**
```python
def get_total_variants(self):
    return self.variants['Variant'].nunique()

def get_all_variants(self):
    return self.variants['Variant'].unique()
```

**Polars Migration:**
```python
def get_total_variants(self):
    return self.variants['Variant'].n_unique()

def get_all_variants(self):
    return self.variants['Variant'].unique().to_list()
```

---

## Row-wise Updates

### Example 6: agent.py - Adding Single Row

**Current Pandas Code:**
```python
def add_action(self, action):
    if action not in self.actions['Name'].values:
        self.actions.loc[len(self.actions)] = [action, 0, 0, 0]
```

**Polars Migration:**
```python
def add_action(self, action):
    if action not in self.actions['Name'].to_list():
        new_row = pl.DataFrame({
            'Name': [action],
            'ActionAttempts': [0.0],
            'ValueEstimates': [0.0],
            'Q': [0.0]
        })
        self.actions = pl.concat([self.actions, new_row], how="vertical")
```

**Alternative - More efficient with batch operations:**
```python
def add_actions(self, actions):
    """Add multiple actions at once - more efficient than one at a time"""
    existing_actions = set(self.actions['Name'].to_list())
    new_actions = [a for a in actions if a not in existing_actions]
    
    if new_actions:
        new_rows = pl.DataFrame({
            'Name': new_actions,
            'ActionAttempts': [0.0] * len(new_actions),
            'ValueEstimates': [0.0] * len(new_actions),
            'Q': [0.0] * len(new_actions)
        })
        self.actions = pl.concat([self.actions, new_rows], how="vertical")
```

---

## String Operations

### Example 7: scenarios.py - String Replacement with Regex

**Current Pandas Code:**
```python
df["Variant"] = df["Variant"].str.replace(r'[!#$%^&*()[]{};:,.<>?|`~=+]', '_', regex=True)
```

**Polars Migration:**
```python
df = df.with_columns([
    pl.col("Variant").str.replace_all(r'[!#$%^&*()[]{};:,.<>?|`~=+]', '_')
])
```

**Note:** Polars string operations:
- `.str.replace()` - replaces first occurrence
- `.str.replace_all()` - replaces all occurrences
- Both support regex by default

---

## Merge and Join Operations

### Example 8: scenarios.py - Merge with Left Join

**Current Pandas Code:**
```python
def _merge_context_features(self, build_df):
    # ...
    merged_df = build_df[["Name"] + list(current_features)].copy()
    
    if previous_features:
        previous_data = previous_build_df[["Name"] + list(previous_features)]
        merged_df = merged_df.merge(previous_data, on="Name", how="left")
    
    # Fill missing values
    feature_means = previous_build_df[list(previous_features)].mean().to_dict()
    for feature in previous_features:
        merged_df[feature] = merged_df[feature].fillna(feature_means.get(feature, 0))
    
    return merged_df
```

**Polars Migration:**
```python
def _merge_context_features(self, build_df):
    # ...
    merged_df = build_df.select(["Name"] + list(current_features))
    
    if previous_features:
        previous_data = previous_build_df.select(["Name"] + list(previous_features))
        merged_df = merged_df.join(previous_data, on="Name", how="left")
    
    # Fill missing values - more efficient in Polars
    feature_means = previous_build_df.select(previous_features).mean()
    
    fill_exprs = []
    for i, feature in enumerate(previous_features):
        mean_val = feature_means[feature][0] if feature_means.height > 0 else 0
        fill_exprs.append(
            pl.col(feature).fill_null(mean_val).alias(feature)
        )
    
    if fill_exprs:
        merged_df = merged_df.with_columns(fill_exprs)
    
    return merged_df
```

---

## Null Handling

### Example 9: agent.py - fillna Operations

**Current Pandas Code:**
```python
def choose(self) -> List[str]:
    if self.t == 0:
        self.last_prioritization = random.sample(self.actions['Name'].tolist(), len(self.actions))
    else:
        # To avoid arms non-applied yet
        self.actions['Q'] = self.actions['Q'].fillna(value=0)
        self.last_prioritization = self.policy.choose_all(self)
    
    return self.last_prioritization
```

**Polars Migration:**
```python
def choose(self) -> List[str]:
    if self.t == 0:
        self.last_prioritization = random.sample(
            self.actions['Name'].to_list(), 
            self.actions.height
        )
    else:
        # To avoid arms non-applied yet
        self.actions = self.actions.with_columns([
            pl.col('Q').fill_null(0.0)
        ])
        self.last_prioritization = self.policy.choose_all(self)
    
    return self.last_prioritization
```

---

### Example 10: scenarios.py - Numeric Conversion with Error Handling

**Current Pandas Code:**
```python
def _read_testcases(self, tcfile: str) -> pd.DataFrame:
    df = pd.read_csv(tcfile, sep=';', parse_dates=['LastRun'], dayfirst=True)
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(0)
    return df
```

**Polars Migration:**
```python
def _read_testcases(self, tcfile: str) -> pl.DataFrame:
    df = pl.read_csv(tcfile, separator=';', try_parse_dates=True)
    
    # Convert to numeric with error handling
    df = df.with_columns([
        pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0.0)
    ])
    
    return df
```

**Note on dayfirst parameter:**
The `dayfirst=True` parameter doesn't have a direct equivalent in Polars. Solutions:

1. **Pre-format dates in data** (recommended)
2. **Use custom parsing:**
```python
def _read_testcases(self, tcfile: str) -> pl.DataFrame:
    df = pl.read_csv(tcfile, separator=';')
    
    # Custom date parsing for DD-MM-YYYY format
    df = df.with_columns([
        pl.col("LastRun").str.strptime(pl.Datetime, format="%d-%m-%Y %H:%M")
    ])
    
    df = df.with_columns([
        pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0.0)
    ])
    
    return df
```

---

## Performance Tips

### Tip 1: Batch Operations Instead of Loops

**Slow (row-by-row):**
```python
for test_case, r in zip(self.last_prioritization, reward):
    self.actions = self.actions.with_columns([...])  # Creating new DF each iteration
```

**Fast (batch operation):**
```python
# Create mapping once
reward_map = dict(zip(self.last_prioritization, reward))

# Single update operation
self.actions = self.actions.with_columns([
    pl.col('ValueEstimates').replace(reward_map, default=pl.col('ValueEstimates'))
])
```

---

### Tip 2: Use Lazy Evaluation for Complex Queries

**Eager:**
```python
df = pl.read_csv("data.csv")
df = df.filter(pl.col('value') > 10)
df = df.select(['Name', 'Value'])
result = df.collect()
```

**Lazy (optimized):**
```python
result = (
    pl.scan_csv("data.csv")
    .filter(pl.col('value') > 10)
    .select(['Name', 'Value'])
    .collect()  # Execute optimized query plan
)
```

---

### Tip 3: Schema Definition for Better Performance

**Without schema (slower, inferred):**
```python
df = pl.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
```

**With schema (faster, explicit):**
```python
schema = {'col1': pl.Int64, 'col2': pl.Utf8}
df = pl.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}, schema=schema)
```

---

### Tip 4: Avoid Repeated to_list() or to_numpy() Calls

**Inefficient:**
```python
for item in df['column'].to_list():
    if item in df['other_column'].to_list():  # to_list() called many times
        # ...
```

**Efficient:**
```python
column_list = df['column'].to_list()
other_column_set = set(df['other_column'].to_list())  # Convert to set once

for item in column_list:
    if item in other_column_set:
        # ...
```

---

## Special Cases and Gotchas

### Gotcha 1: Polars is More Strict with Types

**Pandas (permissive):**
```python
df['int_col'] = df['int_col'] + df['float_col']  # Auto-converts
```

**Polars (strict):**
```python
# Need explicit casting
df = df.with_columns([
    (pl.col('int_col').cast(pl.Float64) + pl.col('float_col')).alias('result')
])
```

---

### Gotcha 2: No Index in Polars

**Pandas:**
```python
df.loc[5] = [1, 2, 3]  # Uses index
df.reset_index(drop=True)
```

**Polars:**
```python
# No index concept - use row numbers or filters
# To get row 5:
row = df[5]  # or df.slice(5, 1)

# No need to reset index - doesn't exist
```

---

### Gotcha 3: Different Null Semantics

**Pandas:**
```python
import numpy as np
df['col'].fillna(np.nan)  # NaN for missing values
```

**Polars:**
```python
df.with_columns([
    pl.col('col').fill_null(None)  # null for missing values
])
```

---

### Gotcha 4: inplace=True Doesn't Exist

**Pandas:**
```python
df.sort_values(by='col', inplace=True)
df.drop_duplicates(inplace=True)
```

**Polars:**
```python
# Always returns new DataFrame
df = df.sort('col')
df = df.unique()
```

---

## Testing Considerations

### Assert DataFrame Equality

**Pandas:**
```python
import pandas.testing as pdt
pdt.assert_frame_equal(df1, df2)
```

**Polars:**
```python
assert df1.frame_equal(df2)

# Or with more control:
assert df1.shape == df2.shape
assert df1.columns == df2.columns
assert all(df1[col].series_equal(df2[col]) for col in df1.columns)
```

---

### Handling Test Fixtures

**Before (Pandas):**
```python
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Name': ['TC1', 'TC2'],
        'Duration': [10, 20]
    })
```

**After (Polars):**
```python
@pytest.fixture
def sample_data():
    return pl.DataFrame({
        'Name': ['TC1', 'TC2'],
        'Duration': [10, 20]
    })
```

---

## Summary of Key Differences

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Copy | `.copy()` | `.clone()` |
| Empty check | `.empty` | `.is_empty()` or `.height == 0` |
| Row count | `len(df)` | `df.height` or `len(df)` |
| Column count | `len(df.columns)` | `df.width` |
| To list | `.tolist()` or `.values` | `.to_list()` or `.to_numpy()` |
| To dict | `.to_dict('records')` | `.to_dicts()` |
| Filter | `df[df['col'] == val]` | `df.filter(pl.col('col') == val)` |
| Select | `df[['col1', 'col2']]` | `df.select(['col1', 'col2'])` |
| Concat | `pd.concat([...], ignore_index=True)` | `pl.concat([...], how="vertical")` |
| Merge | `.merge()` | `.join()` |
| Fill NA | `.fillna()` | `.fill_null()` |
| Unique count | `.nunique()` | `.n_unique()` |
| Sort | `.sort_values()` | `.sort()` |

---

## Recommended Migration Order for Coleman4HCS

1. **Start with utils/monitor.py** - Simplest DataFrame usage
2. **Then scenarios.py** - CSV reading and basic operations  
3. **Then bandit.py** - Similar patterns to agent.py
4. **Then agent.py** - More complex operations
5. **Then policy.py** - Build on agent.py patterns
6. **Finally main.py and statistics** - Integrate everything

This order allows learning Polars patterns incrementally and testing as you go.
