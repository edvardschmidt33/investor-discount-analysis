import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# -----------------------------
# 1. Load data
# -----------------------------
csv_path = "data/investor_preprocess.csv"  # <-- change this
df = pd.read_csv(csv_path, parse_dates=['DATE'])

# Keep years 2016–2024 just in case
df = df[(df['DATE'].dt.year >= 2016) & (df['DATE'].dt.year <= 2024)].copy()

# -----------------------------
# 2. Discretize into low / medium / high
# -----------------------------
def make_qcut_bins(df, col, labels=('low', 'medium', 'high')):
    """Quantile-based bins: roughly 1/3 low, 1/3 medium, 1/3 high."""
    s = df[col]
    valid = s.dropna()
    binned = pd.qcut(valid, q=3, labels=labels, duplicates='drop')
    df[col + '_CAT'] = pd.NA
    df.loc[valid.index, col + '_CAT'] = binned
    return df

# Bin RETURN - OMXS and DISCOUNT/PREMIUM_ADJ
df = make_qcut_bins(df, 'RETURN - OMXS')
df = make_qcut_bins(df, 'DISCOUNT/PREMIUM_ADJ')

# Drop rows where either bin is missing
df = df.dropna(subset=['RETURN - OMXS_CAT', 'DISCOUNT/PREMIUM_ADJ_CAT'])

# -----------------------------
# 3. One-hot encode (basket)
# -----------------------------
cat_df = df[['RETURN - OMXS_CAT', 'DISCOUNT/PREMIUM_ADJ_CAT']].astype('category')

basket = pd.get_dummies(cat_df)

# Rename columns to nice item names (6 items total)
basket = basket.rename(columns={
    'RETURN - OMXS_CAT_low': 'RET_OMXS_low',
    'RETURN - OMXS_CAT_medium': 'RET_OMXS_med',
    'RETURN - OMXS_CAT_high': 'RET_OMXS_high',
    'DISCOUNT/PREMIUM_ADJ_CAT_low': 'DISC_PREM_low',    # = high discount (more negative)
    'DISCOUNT/PREMIUM_ADJ_CAT_medium': 'DISC_PREM_med',
    'DISCOUNT/PREMIUM_ADJ_CAT_high': 'DISC_PREM_high',  # = high premium (more positive)
})

# Sanity check (optional)
# print(basket.head())
# print(basket.sum())

# -----------------------------
# 4. Run FP-growth
# -----------------------------
min_support = 0.03  # tune if needed

frequent_itemsets = fpgrowth(
    basket,
    min_support=min_support,
    use_colnames=True
)

# -----------------------------
# 5. Association rules
# -----------------------------
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0  # only rules with lift >= 1
)

# -----------------------------
# 6. Filter to cross-variable pairs (discount ↔ return-omxs)
#    i.e., 1 item from discount, 1 from return
# -----------------------------
discount_items = {'DISC_PREM_low', 'DISC_PREM_med', 'DISC_PREM_high'}
return_items = {'RET_OMXS_low', 'RET_OMXS_med', 'RET_OMXS_high'}

def is_cross_pair(row):
    ants = row['antecedents']
    cons = row['consequents']
    if len(ants) != 1 or len(cons) != 1:
        return False
    a = next(iter(ants))
    c = next(iter(cons))
    return ((a in discount_items and c in return_items) or
            (a in return_items and c in discount_items))

cross_rules = rules[rules.apply(is_cross_pair, axis=1)].copy()

# Sort by lift and take top 5
cross_rules = cross_rules.sort_values('lift', ascending=False)
top20 = cross_rules.sort_values('lift', ascending=False).head(20)

# Make output readable
top20['antecedents'] = top20['antecedents'].apply(lambda s: next(iter(s)))
top20['consequents'] = top20['consequents'].apply(lambda s: next(iter(s)))

print("Top 20 discount ↔ RETURN-OMXS rules by lift:")
print(top20[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))
# -----------------------------
# 7. Specifically: high discount with high RETURN-OMXS
#    (high discount ≈ DISC_PREM_low, high RETURN-OMXS ≈ RET_OMXS_high)
# -----------------------------
# Look for rule DISC_PREM_low -> RET_OMXS_high
hd_hr_rule = cross_rules[
    (cross_rules['antecedents'] == {'DISC_PREM_low'}) &
    (cross_rules['consequents'] == {'RET_OMXS_high'})
]

if not hd_hr_rule.empty:
    r = hd_hr_rule.iloc[0]
    print("\nRule: high discount (DISC_PREM_low) => high RETURN-OMXS (RET_OMXS_high)")
    print(f"Support   : {r['support']:.4f}")
    print(f"Confidence: {r['confidence']:.4f}")
    print(f"Lift      : {r['lift']:.4f}")
else:
    print("\nNo direct rule DISC_PREM_low => RET_OMXS_high found with current min_support.")
