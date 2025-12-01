"""
Preprocessing Knowledge Base for RAG Agent
Contains best practices and recommendations
"""

PREPROCESSING_KNOWLEDGE = {
    "missing_values": {
        "numeric": {
            "mean_imputation": {
                "when": "Data is normally distributed, no significant outliers",
                "pros": "Simple, fast, maintains mean",
                "cons": "Reduces variance, not robust to outliers",
                "code": "from sklearn.impute import SimpleImputer\nimputer = SimpleImputer(strategy='mean')"
            },
            "median_imputation": {
                "when": "Data has outliers or is skewed",
                "pros": "Robust to outliers, good for skewed data",
                "cons": "May not preserve mean",
                "code": "from sklearn.impute import SimpleImputer\nimputer = SimpleImputer(strategy='median')"
            },
            "knn_imputation": {
                "when": "Features are correlated, dataset not too large",
                "pros": "Uses feature relationships, more accurate",
                "cons": "Computationally expensive, sensitive to scale",
                "code": "from sklearn.impute import KNNImputer\nimputer = KNNImputer(n_neighbors=5)"
            }
        },
        "categorical": {
            "mode_imputation": {
                "when": "Missing values < 5%, categorical data",
                "pros": "Simple, maintains distribution",
                "cons": "Introduces bias, ignores relationships",
                "code": "from sklearn.impute import SimpleImputer\nimputer = SimpleImputer(strategy='most_frequent')"
            },
            "missing_indicator": {
                "when": "Missingness is informative",
                "pros": "Captures missing patterns",
                "cons": "Increases dimensionality",
                "code": "Add new column: df['is_missing'] = df['col'].isna().astype(int)"
            }
        }
    },

    "outliers": {
        "iqr_method": {
            "when": "Data roughly follows normal distribution",
            "threshold": "1.5 for moderate, 3.0 for extreme outliers",
            "action": "Cap values at lower_bound and upper_bound",
            "pros": "Robust, standard method",
            "cons": "May cap legitimate values",
            "code": """Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df['col'] = df['col'].clip(lower, upper)"""
        },
        "zscore_method": {
            "when": "Data is normally distributed",
            "threshold": "2 for moderate, 3 for extreme outliers",
            "action": "Remove or cap values beyond threshold standard deviations",
            "pros": "Good for normal distributions",
            "cons": "Assumes normality, affected by outliers",
            "code": """from scipy import stats
z_scores = np.abs(stats.zscore(df['col']))
df = df[z_scores < 3]"""
        },
        "domain_specific": {
            "when": "You have domain knowledge about valid ranges",
            "action": "Cap or remove based on business rules",
            "example": "Age > 120 is invalid, salary < 0 is invalid",
            "pros": "Most accurate, uses domain knowledge",
            "cons": "Requires expert knowledge"
        }
    },

    "scaling": {
        "standardization": {
            "when": "Features on different scales, algorithms assume normal distribution (SVM, Linear Regression)",
            "formula": "z = (x - mean) / std",
            "pros": "Centers data, useful for algorithms assuming normality",
            "cons": "Affected by outliers",
            "code": "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()"
        },
        "normalization": {
            "when": "Need bounded range [0,1], neural networks, distance-based algorithms",
            "formula": "x_scaled = (x - min) / (max - min)",
            "pros": "Bounded range, good for neural nets",
            "cons": "Very sensitive to outliers",
            "code": "from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()"
        },
        "robust_scaling": {
            "when": "Data has many outliers",
            "formula": "Uses median and IQR instead of mean and std",
            "pros": "Robust to outliers",
            "cons": "May not center data at 0",
            "code": "from sklearn.preprocessing import RobustScaler\nscaler = RobustScaler()"
        }
    },

    "encoding": {
        "label_encoding": {
            "when": "Ordinal categorical variables (low, medium, high) or tree-based models",
            "pros": "Simple, doesn't increase dimensionality",
            "cons": "Implies ordering, not suitable for nominal data in linear models",
            "code": "from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()"
        },
        "onehot_encoding": {
            "when": "Nominal categorical variables with <10 categories",
            "pros": "No false ordering, works with linear models",
            "cons": "Increases dimensionality, curse of dimensionality with many categories",
            "code": "pd.get_dummies(df, columns=['category_col'], drop_first=True)"
        },
        "target_encoding": {
            "when": "High cardinality categorical variables",
            "pros": "Reduces dimensionality, captures target relationship",
            "cons": "Risk of overfitting, requires careful cross-validation",
            "code": "# Use category_encoders library\nfrom category_encoders import TargetEncoder"
        }
    },

    "feature_engineering": {
        "polynomial_features": {
            "when": "Suspicion of non-linear relationships",
            "pros": "Captures interactions and non-linearity",
            "cons": "Exponential increase in features",
            "code": "from sklearn.preprocessing import PolynomialFeatures\npoly = PolynomialFeatures(degree=2)"
        },
        "log_transform": {
            "when": "Right-skewed data, exponential relationships",
            "pros": "Reduces skewness, handles heteroscedasticity",
            "cons": "Can't handle zeros or negatives",
            "code": "df['log_col'] = np.log1p(df['col'])  # log1p handles zeros"
        },
        "binning": {
            "when": "Convert continuous to categorical, reduce noise",
            "pros": "Reduces impact of outliers, simplifies relationships",
            "cons": "Loss of information, arbitrary bin selection",
            "code": "pd.cut(df['age'], bins=[0, 18, 30, 50, 100], labels=['child', 'young', 'middle', 'senior'])"
        }
    }
}

def get_recommendations(data_stats: dict) -> dict:
    """
    Get preprocessing recommendations based on data statistics

    Args:
        data_stats: Dictionary with data statistics

    Returns:
        Dictionary with recommendations
    """
    recommendations = {
        "missing_values": [],
        "outliers": [],
        "scaling": [],
        "encoding": [],
        "warnings": []
    }

    # Missing values recommendations
    if data_stats.get('missing_percentage', 0) > 0:
        if data_stats.get('has_outliers', False):
            recommendations['missing_values'].append({
                "method": "median_imputation",
                "reason": "Data has outliers, median is more robust than mean",
                "details": PREPROCESSING_KNOWLEDGE['missing_values']['numeric']['median_imputation']
            })
        else:
            recommendations['missing_values'].append({
                "method": "mean_imputation",
                "reason": "No significant outliers detected, mean imputation is suitable",
                "details": PREPROCESSING_KNOWLEDGE['missing_values']['numeric']['mean_imputation']
            })

    # Outlier recommendations
    outlier_pct = data_stats.get('outlier_percentage', 0)
    if outlier_pct > 5:
        if outlier_pct > 20:
            recommendations['warnings'].append(
                f"High outlier percentage ({outlier_pct:.1f}%). Consider domain expertise to determine if these are errors or valid extreme values."
            )
        recommendations['outliers'].append({
            "method": "iqr_method",
            "reason": "Standard and robust method for outlier detection",
            "details": PREPROCESSING_KNOWLEDGE['outliers']['iqr_method']
        })

    # Scaling recommendations
    if data_stats.get('scale_variance', 0) > 10:  # High variance in scales
        if outlier_pct > 10:
            recommendations['scaling'].append({
                "method": "robust_scaling",
                "reason": "High variance and outliers present, RobustScaler is recommended",
                "details": PREPROCESSING_KNOWLEDGE['scaling']['robust_scaling']
            })
        else:
            recommendations['scaling'].append({
                "method": "standardization",
                "reason": "Features on different scales, standardization recommended",
                "details": PREPROCESSING_KNOWLEDGE['scaling']['standardization']
            })

    # Encoding recommendations
    if data_stats.get('num_categorical_cols', 0) > 0:
        max_categories = data_stats.get('max_unique_categories', 0)
        if max_categories > 10:
            recommendations['encoding'].append({
                "method": "target_encoding",
                "reason": f"High cardinality categorical variable (max {max_categories} categories)",
                "details": PREPROCESSING_KNOWLEDGE['encoding']['target_encoding']
            })
        else:
            recommendations['encoding'].append({
                "method": "onehot_encoding",
                "reason": f"Low cardinality categorical variables (<10 categories)",
                "details": PREPROCESSING_KNOWLEDGE['encoding']['onehot_encoding']
            })

    return recommendations
