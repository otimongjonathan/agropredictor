# Multiplier Verification: Current Values vs Real-World Uganda Data

## ⚠️ CRITICAL FINDINGS

Based on research, there are **significant discrepancies** between our current multipliers and actual smallholder farmer practices in Uganda.

## Seed Rates Comparison

| Crop | Our Model | Real-World Uganda* | Status |
|------|-----------|-------------------|--------|
| **Maize** | 22.5 kg/acre | **3 kg/acre** | ❌ **7.5x too high** |
| **Beans** | 45 kg/acre | **3 kg/acre** | ❌ **15x too high** |
| **Groundnuts** | 52 kg/acre | **30 kg/acre** | ⚠️ 1.7x too high |
| **Soybeans** | 38 kg/acre | **30 kg/acre** | ⚠️ 1.3x too high |
| **Rice** | 32 kg/acre | Not found | ✅ Likely reasonable |
| **Sorghum** | 8 kg/acre | Not found | ✅ Likely reasonable |

*Source: ISSD Uganda Brief (issduganda.org)

## Fertilizer Usage: MAJOR DISCREPANCY ⚠️

### Uganda Reality:
- **Actual usage**: 2-3 kg per **hectare** = **0.8-1.2 kg per acre**
- Global average: 100 kg/hectare = ~40 kg/acre
- Our model: 80-280 kg/acre

### Our Current Values vs Reality:

| Crop | Our Model | Real Uganda* | Status |
|------|-----------|--------------|--------|
| **Maize** | 125 kg/acre | **0.8-1.2 kg/acre** | ❌ **100-150x too high** |
| **Rice** | 180 kg/acre | **0.8-1.2 kg/acre** | ❌ **150-225x too high** |
| **Beans** | 80 kg/acre | **0.8-1.2 kg/acre** | ❌ **67-100x too high** |
| **Coffee** | 280 kg/acre | **0.8-1.2 kg/acre** | ❌ **233-350x too high** |
| **All crops** | 80-280 kg/acre | **0.8-1.2 kg/acre** | ❌ **Severely overestimated** |

*Source: Economics Observatory / Uganda Agriculture Ministry

**This is a CRITICAL issue!** Uganda smallholder farmers use almost NO fertilizer compared to global standards.

## Labor Days: Likely Reasonable ✅

Our labor day estimates (14-38 days/acre) seem reasonable for smallholder farming practices.

## Chemical Inputs: Unknown

Herbicide and pesticide usage rates in Uganda are not well-documented. Our values (1-3.5 L/acre) may be reasonable but need verification.

## Recommendations

### 1. **URGENT: Adjust Seed Rates**
- **Maize**: 22.5 → **3 kg/acre**
- **Beans**: 45 → **3 kg/acre**  
- **Groundnuts**: 52 → **30 kg/acre**
- **Soybeans**: 38 → **30 kg/acre**

### 2. **CRITICAL: Adjust Fertilizer Rates**
Since Uganda farmers use almost no fertilizer (0.8-1.2 kg/acre), we have two options:

**Option A: Realistic Smallholder Values** (Recommended)
- All crops: **1-2 kg/acre** (reflecting actual usage)
- Pros: Accurate for smallholders
- Cons: May not reflect "best practice" costs

**Option B: Hybrid Approach**
- Use low values (5-20 kg/acre) for basic crops
- Use higher values (50-100 kg/acre) for commercial/cash crops
- Reflects that some farmers invest more

**Option C: Provide Both Views**
- Add toggle: "Smallholder Mode" vs "Commercial/Best Practice Mode"
- Smallholder: 1-2 kg/acre
- Commercial: Current FAO-based values

### 3. **Verify Chemical Rates**
- Need local data on herbicide/pesticide usage
- Current values may be reasonable but unverified

### 4. **Crop-Specific Adjustments**
Based on research:
- **Rice**: Can earn 1.2M UGX/acre at 4 tonnes/hectare yield
- **Beans**: Average yield 131 kg/acre (below potential 250 kg/acre)
- **Groundnuts**: Average 321 kg/acre (potential 1,200 kg/acre)

## Next Steps

1. ✅ Update seed rates to match Uganda reality
2. ⚠️ **CRITICAL**: Decide on fertilizer strategy (realistic vs best practice)
3. ⚠️ Verify chemical input rates with local sources
4. 📊 Add note in UI about smallholder vs commercial practices

## Sources

1. ISSD Uganda - Seed rates and costs
2. Economics Observatory - Fertilizer usage in Uganda
3. Uganda Ministry of Agriculture reports
4. FAO guidelines (for best practices reference)

## Impact Assessment

**Current Model Issues:**
- Predictions likely **overestimate** input costs significantly
- May not reflect actual smallholder economics
- Fertilizer costs are **dramatically inflated** (100-300x)

**User Impact:**
- Predictions may scare away smallholder farmers
- Costs don't match their actual input usage
- Model appears unrealistic for Uganda context

---

**Recommendation**: Update multipliers to reflect REAL smallholder practices in Uganda, not ideal/global standards.

