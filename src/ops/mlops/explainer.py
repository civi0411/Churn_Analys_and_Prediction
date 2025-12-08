"""
src/ops/mlops/explainer.py

Model Explainer - SHAP and Feature Importance.
"""
import pandas as pd
from typing import List, Optional


class ModelExplainer:
    """Model Explainability - SHAP and Feature Importance."""

    def __init__(self, model, X_train, feature_names: List[str], logger=None):
        self.logger = logger

        # Unwrap model from pipeline if needed
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            self.model = model.named_steps["model"]
        else:
            self.model = model

        self.X_train = X_train
        self.feature_names = feature_names
        # Precompute numeric medians for training columns (coerce bracketed/scientific strings)
        self._train_medians = {}
        try:
            for col in getattr(self.X_train, 'columns', []):
                try:
                    series = self.X_train[col].astype(str).str.strip().str.strip('[]()')
                    numeric = pd.to_numeric(series, errors='coerce')
                    med = numeric.median()
                    if pd.isna(med):
                        med = 0.0
                except Exception:
                    med = 0.0
                self._train_medians[col] = med
        except Exception:
            self._train_medians = {}

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Get feature importance from model."""
        if not hasattr(self.model, "feature_importances_"):
            if self.logger:
                self.logger.warning("Model does not have feature_importances_")
            return None

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).head(top_n)

        return importance_df

    def explain_with_shap(self, X_sample, save_path: str = None) -> bool:
        """Generate SHAP explanations."""
        try:
            import shap
            import matplotlib.pyplot as plt
            import re
            import warnings

            if self.logger:
                self.logger.info("Generating SHAP explanations...")

            shap_values = None
            X_sample_clean = None

            def _try_numeric_coerce(df):
                dfc = df.copy()
                for col in dfc.select_dtypes(include=['object', 'string']).columns:
                    def _coerce(val):
                        if isinstance(val, str):
                            # strip surrounding brackets/spaces
                            s = val.strip().strip('[]()')
                            # match numeric pattern (including scientific notation)
                            m = re.match(r"^[+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?$", s)
                            if m:
                                try:
                                    return float(s)
                                except Exception:
                                    return val
                            else:
                                return val
                        return val

                    try:
                        dfc[col] = dfc[col].apply(_coerce)
                        # if column now numeric, cast
                        if pd.to_numeric(dfc[col], errors='coerce').notna().all():
                            dfc[col] = pd.to_numeric(dfc[col], errors='coerce')
                    except Exception:
                        # leave column unchanged
                        continue
                return dfc

            # Method 1: TreeExplainer (fastest)
            try:
                X_sample_clean = _try_numeric_coerce(X_sample)

                def _prepare_numeric(df):
                    X_num = df.copy()
                    for col in X_num.columns:
                        X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
                        if X_num[col].isna().any():
                            median_val = self._train_medians.get(col, 0.0)
                            X_num[col] = X_num[col].fillna(median_val)
                    # Keep only numeric columns for TreeExplainer
                    numeric_df = X_num.select_dtypes(include=['number']).copy()
                    import numpy as _np
                    return _np.asarray(numeric_df).astype(float), numeric_df

                # First attempt
                try:
                    X_arr, X_num = _prepare_numeric(X_sample_clean)
                    # If no numeric columns available, raise to trigger fallback
                    if X_num.shape[1] == 0:
                        raise ValueError("No numeric features available for TreeExplainer")
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_arr)
                    if self.logger:
                        self.logger.info("  Using TreeExplainer (initial)")
                except Exception as tree_err_first:
                    # Log the warning so operator can see the root issue
                    if self.logger:
                        try:
                            self.logger.warning(f"TreeExplainer initial attempt failed: {tree_err_first}")
                        except Exception:
                            pass

                    # Aggressive cleaning and retry
                    try:
                        # produce a more aggressively cleaned numeric DataFrame from raw X_sample
                        X_sample_more = X_sample.copy()
                        for col in X_sample_more.select_dtypes(include=['object', 'string']).columns:
                            X_sample_more[col] = X_sample_more[col].astype(str).str.strip().str.strip('[]()')
                            X_sample_more[col] = pd.to_numeric(X_sample_more[col], errors='coerce')
                        # fill NA with training medians
                        for col in X_sample_more.columns:
                            if X_sample_more[col].isna().any():
                                X_sample_more[col] = X_sample_more[col].fillna(self._train_medians.get(col, 0.0))
                        # After aggressive cleaning, select numeric columns only
                        X_sample_more_num = X_sample_more.select_dtypes(include=['number']).copy()
                        if X_sample_more_num.shape[1] == 0:
                            raise ValueError("No numeric features after aggressive clean")
                        X_arr2 = __import__('numpy').array(X_sample_more_num.astype(float))
                        explainer = shap.TreeExplainer(self.model)
                        shap_values = explainer.shap_values(X_arr2)
                        if self.logger:
                            self.logger.info("  Using TreeExplainer (after aggressive clean)")
                    except Exception as tree_err_second:
                        # If still failing, fallback to KernelExplainer (log info about fallback)
                        if self.logger:
                            try:
                                self.logger.info(f"TreeExplainer retry failed; falling back to KernelExplainer: {tree_err_second}")
                            except Exception:
                                pass
                        raise tree_err_second
            except Exception as tree_error:
                if self.logger:
                    self.logger.warning(f"  TreeExplainer failed: {tree_error}")

                # Method 2: KernelExplainer (slower but compatible)
                try:
                    def model_predict(x):
                        return self.model.predict_proba(x)[:, 1]

                    # Try to coerce X_sample to numeric array for KernelExplainer
                    import numpy as _np
                    try:
                        X_sample_num = X_sample.copy()
                        # convert any string-numeric like '[5E-1]' to float if possible and keep numeric cols only
                        for col in X_sample_num.select_dtypes(include=['object', 'string']).columns:
                            try:
                                X_sample_num[col] = pd.to_numeric(X_sample_num[col].astype(str).str.strip().str.strip('[]()'), errors='coerce')
                            except Exception:
                                # leave as is
                                pass
                        # select numeric columns only for Kernel background
                        X_sample_num = X_sample_num.select_dtypes(include=['number']).copy()
                        if X_sample_num.shape[1] == 0:
                            # fallback to original if no numeric columns
                            X_sample_num = X_sample.copy()
                    except Exception:
                        X_sample_num = X_sample

                    # Use an explicit int seed for shap.sample for compatibility
                    try:
                        background = shap.sample(X_sample_num, min(50, len(X_sample_num)), random_state=42)
                    except TypeError:
                        # Older shap versions may not accept random_state kwarg; fallback
                        background = shap.sample(X_sample_num, min(50, len(X_sample_num)))

                    explainer = shap.KernelExplainer(model_predict, background)
                    shap_values = explainer.shap_values(X_sample_num, nsamples=100)
                    if self.logger:
                        self.logger.info("  Using KernelExplainer (slower but compatible)")
                except Exception as kernel_error:
                    if self.logger:
                        self.logger.warning(f"  KernelExplainer also failed: {kernel_error}")
                    return False

            if shap_values is None:
                if self.logger:
                    self.logger.warning("  Could not compute SHAP values")
                return False

            # Plot - suppress known FutureWarning from SHAP about RNG seeding
            plt.figure(figsize=(10, 6))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*NumPy global RNG was seeded.*", category=FutureWarning)
                # Use the cleaned sample for plotting if available
                try:
                    plot_sample = X_sample_clean
                except NameError:
                    plot_sample = X_sample
                shap.summary_plot(
                    shap_values, plot_sample,
                    feature_names=self.feature_names, show=False
                )

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=150)
                plt.close()
                if self.logger:
                    self.logger.info(f"SHAP Plot     | {save_path}")
                return True
            else:
                plt.show()
                return True

        except ImportError:
            msg = "SHAP not installed. Run: pip install shap"
            if self.logger:
                self.logger.warning(f"[SHAP] {msg}")
            else:
                print(f"[WARN] {msg}")
            return False

        except Exception as e:
            msg = f"SHAP explanation failed: {e}"
            if self.logger:
                self.logger.warning(f"[SHAP] {msg}")
            else:
                print(f"[ERROR] {msg}")
            return False
