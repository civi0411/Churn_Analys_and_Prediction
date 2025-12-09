import re
from typing import Dict, Any, List

# Helper utilities for reporting module

def sanitize_filename(name: str) -> str:
    """Return a safe filename by replacing non-alphanumeric characters with underscore."""
    # keep alnum, dash, underscore
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", name)
    # collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    return safe.strip('_')


def sanitize_text(text: str) -> str:
    """Remove non-printable or special emoji-like characters to keep ASCII-friendly text.
    This keeps basic punctuation and letters; removes higher unicode symbols like emoji.
    """
    if not isinstance(text, str):
        return text
    # Remove control chars
    text = ''.join(ch for ch in text if ch.isprintable())
    # Remove emojis / non-BMP characters by excluding characters outside basic multilingual plane
    # Keep characters in range U+0000 - U+07FF (covers basic Latin and many others)
    text = ''.join(ch for ch in text if ord(ch) <= 0x07FF)
    return text


def figure_insights(fname: str, best_metrics: Dict[str, float], all_metrics: Dict[str, Dict[str, float]], feature_importance: Any) -> List[str]:
    """Re-usable figure insights generator (same logic as before)."""
    notes: List[str] = []
    lname = fname.lower()
    try:
        if 'roc' in lname:
            auc = best_metrics.get('roc_auc') if best_metrics else None
            if auc:
                notes.append(f"- ROC-AUC (best model): **{auc:.4f}** — indicates the model's discrimination ability (closer to 1 is better).")
            else:
                notes.append("- ROC curve available — review AUC to assess separability.")

        elif 'confusion' in lname:
            rec = best_metrics.get('recall') if best_metrics else None
            prec = best_metrics.get('precision') if best_metrics else None
            f1 = best_metrics.get('f1') if best_metrics else None
            if rec is not None:
                notes.append(f"- Recall (Churn coverage): **{rec:.4f}** — proportion of churn instances detected.")
            if prec is not None:
                notes.append(f"- Precision: **{prec:.4f}** — among predicted churn, proportion correct.")
            if f1 is not None:
                notes.append(f"- F1-score: **{f1:.4f}** — harmonic mean of precision & recall.")
            notes.append("- Actionable: prioritize reducing False Negatives (increase recall) for retention campaigns.")

        elif 'feature_importance' in lname or 'importance' in lname:
            lines = []
            if feature_importance is not None:
                try:
                    if hasattr(feature_importance, 'head'):
                        top = feature_importance.head(5)
                        for _, r in top.iterrows():
                            lines.append(f"{r['feature']}: {r['importance']:.4f}")
                    elif isinstance(feature_importance, dict):
                        for k, v in list(feature_importance.items())[:5]:
                            lines.append(f"{k}: {v:.4f}")
                except Exception:
                    pass
            if lines:
                notes.append("- Top features (preview):")
                for l in lines:
                    notes.append(f"  - {l}")
                notes.append("- Business insight: address top drivers (e.g., Tenure, Complain) in retention playbooks.")
            else:
                notes.append("- Feature importance plot available; review top features impacting predictions.")

        elif 'model_comparison' in lname or 'comparison' in lname:
            try:
                sorted_models = sorted(all_metrics.items(), key=lambda kv: kv[1].get('f1', 0), reverse=True)
                if sorted_models:
                    top_name, top_metrics = sorted_models[0]
                    notes.append(f"- Top model by F1: **{top_name}** (F1={top_metrics.get('f1',0):.4f}).")
                    if len(sorted_models) > 1:
                        second_name, second_metrics = sorted_models[1]
                        notes.append(f"- Gap to 2nd ({second_name}): {top_metrics.get('f1',0)-second_metrics.get('f1',0):.4f} in F1.")
            except Exception:
                pass

        elif 'missing' in lname:
            notes.append("- Shows columns with missing values; consider imputation (median for numeric, mode for categorical) before modeling.")
        elif 'target_distribution' in lname or 'distribution' in lname:
            notes.append("- Shows class distribution; if highly imbalanced, consider resampling (SMOTE/Tomek) or class-weighting.")
        elif 'correlation' in lname:
            notes.append("- Correlation matrix highlights multicollinearity and features correlated with target — consider feature selection.")
        else:
            notes.append("- Figure available: review visual for domain insights.")
    except Exception:
        notes.append("- Unable to auto-generate figure insights; review manually.")
    return notes

