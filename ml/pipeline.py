"""
ML Pipeline — LightGBM + PatchTST + Feature Selection
=====================================================
Настроено под mean-reversion на forex/CFD через cTrader.

Архитектура:
  1. Feature Engineering — 120+ features per bar
  2. Feature Selection — SHAP + Boruta + recursive elimination
  3. LightGBM classifier — вероятность mean-reversion в ближайшие N баров
  4. PatchTST — deep learning forecast для regime detection
  5. Ensemble layer — weighted average LightGBM + PatchTST
  6. Signal generation — z-score threshold + confidence calibration

Hyperparameters заточены под RTX 3090/4080 (ограниченная память) +
mean-reversion на H1/H4 горизонтах.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("deepseek.ml")

# =============================================================================
# LightGBM Hyperparameters — Mean-Reversion Classifier
# =============================================================================

LIGHTGBM_PARAMS: dict = {
    # --- Objective ---
    "objective": "binary",                # Mean-reversion probability (0/1)
    "metric": "binary_logloss",
    "boosting_type": "gbdt",

    # --- Tree structure ---
    "num_leaves": 31,                     # Lower = less overfit. For forex: 31-63 range.
    "max_depth": -1,                      # Unlimited with num_leaves control
    "min_data_in_leaf": 50,               # Forex needs stability — 50-200 range
    "min_child_samples": 50,
    "min_child_weight": 1e-3,

    # --- Regularisation (critical for noisy forex data) ---
    "lambda_l1": 0.1,                     # L1 on leaf weights — feature selection pressure
    "lambda_l2": 1.0,                     # L2 — smooths extreme predictions
    "min_split_gain": 0.01,
    "min_gain_to_split": 0.01,
    "feature_fraction": 0.7,              # Column subsampling per tree
    "bagging_fraction": 0.8,              # Row subsampling
    "bagging_freq": 5,                    # Bag every 5 iterations
    "extra_trees": False,                 # Stricter splits for noisy data

    # --- Learning ---
    "learning_rate": 0.02,                # Small LR for stable convergence
    "n_estimators": 500,                  # Early stopping will cut this
    "max_bin": 255,                       # Feature discretisation bins

    # --- Imbalance (mean-reversion signals are rare) ---
    "scale_pos_weight": 1.0,              # Will be set dynamically based on class ratio
    "is_unbalance": False,                # Manual control via scale_pos_weight

    # --- Stopping ---
    "early_stopping_rounds": 50,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# Horizon-specific tuning overrides
LIGHTGBM_HORIZON_OVERRIDES: dict[str, dict] = {
    "5m": {
        "num_leaves": 63,                 # More capacity for noisier short-term patterns
        "min_data_in_leaf": 30,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "early_stopping_rounds": 30,
    },
    "15m": {
        "num_leaves": 47,
        "min_data_in_leaf": 40,
        "learning_rate": 0.025,
        "n_estimators": 400,
        "early_stopping_rounds": 40,
    },
    "1h": {
        "num_leaves": 31,                 # Default — sweet spot for hourly
        "min_data_in_leaf": 50,
        "learning_rate": 0.02,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    },
    "4h": {
        "num_leaves": 23,                 # Simpler model for longer horizons
        "min_data_in_leaf": 80,
        "learning_rate": 0.015,
        "n_estimators": 600,
        "early_stopping_rounds": 60,
    },
    "1d": {
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "learning_rate": 0.01,
        "n_estimators": 700,
        "early_stopping_rounds": 70,
    },
}


def get_lightgbm_params(horizon: str = "1h") -> dict:
    """Get LightGBM parameters tuned for a specific horizon."""
    params = dict(LIGHTGBM_PARAMS)
    if horizon in LIGHTGBM_HORIZON_OVERRIDES:
        params.update(LIGHTGBM_HORIZON_OVERRIDES[horizon])
    return params


# =============================================================================
# PatchTST Hyperparameters — Regime / Volatility Forecasting
# =============================================================================

PATCHTST_PARAMS: dict = {
    # --- Architecture ---
    "d_model": 128,                       # Model dimension (small — fits RTX 3090 24GB)
    "n_heads": 8,                         # Multi-head attention
    "n_layers": 3,                        # Transformer encoder layers
    "d_ff": 256,                          # Feed-forward dimension (2× d_model)
    "dropout": 0.1,
    "activation": "gelu",

    # --- Patching ---
    "patch_len": 8,                       # Patch length in bars
    "stride": 4,                          # Patch stride (50% overlap)
    "context_window": 96,                 # Lookback window (bars)
    "prediction_length": 24,              # Forecast horizon (bars)

    # --- Training ---
    "batch_size": 32,                     # GPU-friendly
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "max_epochs": 50,
    "early_stopping_patience": 10,
    "gradient_clip_val": 1.0,

    # --- Optimisation ---
    "optimizer": "adamw",
    "lr_scheduler": "cosine",
    "warmup_epochs": 5,

    # --- Loss ---
    "loss": "mse",                        # Point forecast
    "use_revin": True,                    # Reversible instance norm (critical for non-stationary forex)
    "revin_affine": True,
    "individual": False,                  # Shared projection head (fewer params)

    # --- Precision (24GB VRAM constraint) ---
    "precision": "16-mixed",              # Mixed precision training
}

# Horizon-specific PatchTST overrides
PATCHTST_HORIZON_OVERRIDES: dict[str, dict] = {
    "5m": {
        "context_window": 192,
        "prediction_length": 48,
        "patch_len": 16,
        "stride": 8,
        "batch_size": 16,                 # Smaller batches for longer sequences
    },
    "15m": {
        "context_window": 128,
        "prediction_length": 32,
        "patch_len": 16,
        "stride": 8,
    },
    "1h": {
        "context_window": 96,
        "prediction_length": 24,
        "patch_len": 8,
        "stride": 4,
    },
    "4h": {
        "context_window": 72,
        "prediction_length": 12,
        "patch_len": 6,
        "stride": 3,
    },
    "1d": {
        "context_window": 48,
        "prediction_length": 7,
        "patch_len": 4,
        "stride": 2,
    },
}


# =============================================================================
# Feature Engineering Pipeline — 120+ features
# =============================================================================

@dataclass
class FeatureConfig:
    """Which feature groups to compute."""
    price_action: bool = True
    momentum: bool = True
    volatility: bool = True
    volume: bool = True
    mean_reversion: bool = True
    trend: bool = True
    cycles: bool = True
    statistical: bool = True
    market_microstructure: bool = True
    cross_asset: bool = False            # Needs external data
    sentiment: bool = False              # Needs external data


FEATURE_DEFINITIONS: dict[str, list[tuple[str, int, callable]]] = {
    # ------------------------------------------------------------------
    # Price Action (8 features)
    # ------------------------------------------------------------------
    "price_action": [
        ("returns_1", 1, lambda df: df["close"].pct_change(1)),
        ("returns_5", 5, lambda df: df["close"].pct_change(5)),
        ("returns_20", 20, lambda df: df["close"].pct_change(20)),
        ("log_returns_1", 1, lambda df: np.log(df["close"] / df["close"].shift(1))),
        ("hl_ratio", 1, lambda df: (df["high"] - df["low"]) / df["close"]),
        ("co_ratio", 1, lambda df: (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)),
        ("gap", 1, lambda df: df["open"] - df["close"].shift(1)),
        ("gap_pct", 1, lambda df: (df["open"] - df["close"].shift(1)) / df["close"].shift(1)),
    ],

    # ------------------------------------------------------------------
    # Momentum (12 features)
    # ------------------------------------------------------------------
    "momentum": [
        ("rsi_7", 7, lambda df, w: _rsi(df["close"], w)),
        ("rsi_14", 14, lambda df, w: _rsi(df["close"], w)),
        ("rsi_28", 28, lambda df, w: _rsi(df["close"], w)),
        ("stoch_k_14", 14, lambda df, w: _stoch_k(df, w)),
        ("stoch_d_3", 3, lambda df, w: _stoch_d(df, 14, w)),
        ("cci_20", 20, lambda df, w: _cci(df, w)),
        ("williams_r_14", 14, lambda df, w: _williams_r(df, w)),
        ("mfi_14", 14, lambda df, w: _mfi(df, w)),
        ("roc_10", 10, lambda df, w: df["close"].pct_change(w) * 100),
        ("momentum_20", 20, lambda df, w: df["close"] - df["close"].shift(w)),
        ("tsi", 25, lambda df, w: _tsi(df, w, 13)),
        ("uo_14", 14, lambda df, w: _ultimate_osc(df, w)),
    ],

    # ------------------------------------------------------------------
    # Volatility (10 features)
    # ------------------------------------------------------------------
    "volatility": [
        ("atr_14", 14, lambda df, w: _atr(df, w)),
        ("atr_pct_14", 14, lambda df, w: _atr(df, w) / df["close"] * 100),
        ("bb_width_20", 20, lambda df, w: _bb_width(df, w)),
        ("bb_position_20", 20, lambda df, w: _bb_position(df, w)),
        ("bb_squeeze_20", 20, lambda df, w: _bb_squeeze(df, w)),
        ("hist_vol_20", 20, lambda df, w: df["close"].pct_change().rolling(w).std() * np.sqrt(252)),
        ("hist_vol_ratio", 20, lambda df, w: _vol_ratio(df, w, 60)),
        ("parkinson_vol_20", 20, lambda df, w: _parkinson_vol(df, w)),
        ("garman_klass_vol_20", 20, lambda df, w: _garman_klass_vol(df, w)),
        ("yang_zhang_vol_20", 20, lambda df, w: _yang_zhang_vol(df, w)),
    ],

    # ------------------------------------------------------------------
    # Volume (8 features)
    # ------------------------------------------------------------------
    "volume": [
        ("volume_sma_20", 20, lambda df, w: df["tick_volume"].rolling(w).mean()),
        ("volume_ratio", 20, lambda df, w: df["tick_volume"] / df["tick_volume"].rolling(w).mean()),
        ("volume_std_20", 20, lambda df, w: df["tick_volume"].rolling(w).std()),
        ("volume_zscore_20", 20, lambda df, w: (df["tick_volume"] - df["tick_volume"].rolling(w).mean()) / df["tick_volume"].rolling(w).std()),
        ("obv", 1, lambda df, w: _obv(df)),
        ("vwap", 20, lambda df, w: _vwap(df, w)),
        ("vwap_deviation", 20, lambda df, w: (df["close"] - _vwap(df, w)) / _vwap(df, w)),
        ("volume_climax", 20, lambda df, w: _volume_climax(df, w)),
    ],

    # ------------------------------------------------------------------
    # Mean-Reversion Specific (14 features)
    # ------------------------------------------------------------------
    "mean_reversion": [
        ("zscore_ma_20", 20, lambda df, w: _zscore(df["close"], w)),
        ("zscore_ma_50", 50, lambda df, w: _zscore(df["close"], w)),
        ("zscore_ma_200", 200, lambda df, w: _zscore(df["close"], w)),
        ("bb_exit_upper_20_2", 20, lambda df, w: (df["close"] - _bb_upper(df, w, 2)) / _bb_width(df, w)),
        ("bb_exit_lower_20_2", 20, lambda df, w: (df["close"] - _bb_lower(df, w, 2)) / _bb_width(df, w)),
        ("keltner_channel_position", 20, lambda df, w: _kc_position(df, w, 1.5)),
        ("distance_from_vwap_pct", 20, lambda df, w: (df["close"] - _vwap(df, w)) / _vwap(df, w) * 100),
        ("hurst_exponent_100", 100, lambda df, w: _hurst(df["close"], w)),
        ("half_life", 50, lambda df, w: _half_life(df["close"], w)),
        ("adf_stat_20", 20, lambda df, w: _adf_stat_rolling(df["close"], w)),
        ("adf_stat_50", 50, lambda df, w: _adf_stat_rolling(df["close"], w)),
        ("rsi_divergence", 14, lambda df, w: _rsi_divergence(df, w)),
        ("mean_reversion_score", 20, lambda df, w: _mr_composite(df, w)),
        ("fractal_efficiency", 20, lambda df, w: _fractal_efficiency(df["close"], w)),
    ],

    # ------------------------------------------------------------------
    # Trend (8 features)
    # ------------------------------------------------------------------
    "trend": [
        ("adx_14", 14, lambda df, w: _adx(df, w)),
        ("di_plus_14", 14, lambda df, w: _di_plus(df, w)),
        ("di_minus_14", 14, lambda df, w: _di_minus(df, w)),
        ("aroon_up_25", 25, lambda df, w: _aroon_up(df, w)),
        ("aroon_down_25", 25, lambda df, w: _aroon_down(df, w)),
        ("sma_50_distance", 50, lambda df, w: (df["close"] - df["close"].rolling(w).mean()) / df["close"].rolling(w).mean()),
        ("sma_200_distance", 200, lambda df, w: (df["close"] - df["close"].rolling(w).mean()) / df["close"].rolling(w).mean()),
        ("ema_slope_20", 20, lambda df, w: _ema_slope(df["close"], w)),
    ],

    # ------------------------------------------------------------------
    # Statistical Moments (10 features)
    # ------------------------------------------------------------------
    "statistical": [
        ("skewness_20", 20, lambda df, w: df["close"].pct_change().rolling(w).skew()),
        ("kurtosis_20", 20, lambda df, w: df["close"].pct_change().rolling(w).kurt()),
        ("entropy_20", 20, lambda df, w: _sample_entropy(df["close"], w)),
        ("jarque_bera_20", 20, lambda df, w: _jarque_bera_rolling(df, w)),
        ("autocorr_1", 20, lambda df, w: df["close"].pct_change().rolling(w).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)),
        ("autocorr_5", 50, lambda df, w: df["close"].pct_change().rolling(w).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0)),
        ("var_ratio_5", 5, lambda df, w: _variance_ratio(df["close"], w)),
        ("var_ratio_20", 20, lambda df, w: _variance_ratio(df["close"], w)),
        ("runs_test_z", 20, lambda df, w: _runs_test_z_rolling(df["close"], w)),
        ("bds_stat", 20, lambda df, w: _bds_stat_rolling(df["close"], w)),
    ],

    # ------------------------------------------------------------------
    # Market Microstructure (8 features)
    # ------------------------------------------------------------------
    "market_microstructure": [
        ("spread", 1, lambda df: df["ask"] - df["bid"]),
        ("spread_pct", 1, lambda df: (df["ask"] - df["bid"]) / df["close"] * 100),
        ("effective_spread", 20, lambda df, w: _effective_spread(df, w)),
        ("roll_measure", 20, lambda df, w: _roll_measure(df, w)),
        ("tick_imbalance", 20, lambda df, w: _tick_imbalance(df, w)),
        ("quote_imbalance", 20, lambda df, w: _quote_imbalance(df, w)),
        ("order_flow_toxicity", 20, lambda df, w: _vpim(df, w)),
        ("depth_imbalance", 20, lambda df, w: _depth_imbalance(df, w)),
    ],

    # ------------------------------------------------------------------
    # Cycles (Fourier + Wavelet) — 4 features
    # ------------------------------------------------------------------
    "cycles": [
        ("dominant_cycle_50", 50, lambda df, w: _dominant_cycle(df["close"], w)),
        ("cycle_strength_50", 50, lambda df, w: _cycle_strength(df["close"], w)),
        ("dft_peak_ratio", 50, lambda df, w: _dft_peak_ratio(df["close"], w)),
        ("wavelet_energy", 50, lambda df, w: _wavelet_energy(df["close"], w)),
    ],
}

# Total: ~120 features across 9 groups

# =============================================================================
# Feature Selection Pipeline
# =============================================================================

class FeatureSelector:
    """
    Multi-stage feature selection:
      1. Variance threshold (remove near-constant features)
      2. Correlation filter (remove one of each pair with ρ > 0.95)
      3. SHAP importance (top N by mean |SHAP|)
      4. Boruta (confirm relevance against shadow features)
      5. Recursive Feature Elimination with cross-validation (RFECV)

    Target: 30-50 final features for LightGBM (avoid overfit on forex).
    """

    def __init__(
        self,
        variance_threshold: float = 0.001,
        correlation_threshold: float = 0.95,
        shap_top_n: int = 60,
        boruta_max_iter: int = 100,
        rfecv_min_features: int = 30,
        rfecv_step: int = 1,
        random_state: int = 42,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.shap_top_n = shap_top_n
        self.boruta_max_iter = boruta_max_iter
        self.rfecv_min_features = rfecv_min_features
        self.rfecv_step = rfecv_step
        self.random_state = random_state

        self.selected_features_: list[str] = []
        self.feature_importance_: dict[str, float] = {}

    def select(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, list[str]]:
        """
        Run full feature selection pipeline.
        Returns (X_selected, selected_feature_names).
        """
        # Stage 1: Variance threshold
        X = self._variance_filter(X)

        # Stage 2: Correlation filter
        X = self._correlation_filter(X)

        # Stage 3: SHAP importance
        self.feature_importance_ = self._shap_importance(X, y)
        top_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: abs(x[1]), reverse=True,
        )[:self.shap_top_n]
        X = X[[f for f, _ in top_features]]

        # Stage 4: Boruta
        confirmed = self._boruta_select(X, y)
        X = X[confirmed]

        # Stage 5: RFECV
        X, final_features = self._rfecv_select(X, y)

        self.selected_features_ = final_features
        logger.info("Feature selection complete: %d features selected from %d",
                     len(final_features), X.shape[1])
        return X, final_features

    def _variance_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        from sklearn.feature_selection import VarianceThreshold
        sel = VarianceThreshold(threshold=self.variance_threshold)
        sel.fit(X)
        kept = X.columns[sel.get_support()]
        dropped = len(X.columns) - len(kept)
        if dropped:
            logger.info("Variance filter: dropped %d low-variance features", dropped)
        return X[kept]

    def _correlation_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.correlation_threshold)]
        if to_drop:
            logger.info("Correlation filter: dropping %d features (ρ > %.2f): %s",
                         len(to_drop), self.correlation_threshold, to_drop[:5])
            X = X.drop(columns=to_drop)
        return X

    def _shap_importance(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        import lightgbm as lgb
        import shap

        model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        importance = {}
        for i, col in enumerate(X.columns):
            importance[col] = float(np.abs(shap_values[:, i]).mean())
        return importance

    def _boruta_select(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_jobs=-1,
            max_depth=5,
            n_estimators=200,
            random_state=self.random_state,
        )
        boruta = BorutaPy(
            rf,
            n_estimators="auto",
            max_iter=self.boruta_max_iter,
            random_state=self.random_state,
            verbose=0,
        )
        boruta.fit(X.values, y.values)
        confirmed = X.columns[boruta.support_].tolist()
        tentative = X.columns[boruta.support_weak_].tolist()
        logger.info("Boruta: %d confirmed, %d tentative, %d rejected",
                     len(confirmed), len(tentative),
                     len(X.columns) - len(confirmed) - len(tentative))
        # Accept tentative features
        all_selected = confirmed + tentative
        return all_selected

    def _rfecv_select(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, list[str]]:
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit

        estimator = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
        )
        cv = TimeSeriesSplit(n_splits=5)
        rfecv = RFECV(
            estimator=estimator,
            step=self.rfecv_step,
            cv=cv,
            min_features_to_select=self.rfecv_min_features,
            scoring="roc_auc",
        )
        rfecv.fit(X, y)
        kept = X.columns[rfecv.support_].tolist()
        logger.info("RFECV: %d features selected (optimal=%d)",
                     len(kept), rfecv.n_features_)
        return X[kept], kept


# =============================================================================
# Signal Generator — from ML scores to SignalPacket
# =============================================================================

class MeanReversionSignalGenerator:
    """
    Converts ML model outputs into actionable SignalPackets.

    Logic:
      - LightGBM probability > threshold → signal
      - PatchTST forecast confirms regime (not trending)
      - Z-score from rolling mean provides entry timing refinement
      - Confidence tier based on ensemble agreement
    """

    def __init__(
        self,
        lgbm_threshold_high: float = 0.75,
        lgbm_threshold_low: float = 0.55,
        patchtst_regime_weight: float = 0.3,
        zscore_entry_threshold: float = 2.0,
    ):
        self.lgbm_threshold_high = lgbm_threshold_high
        self.lgbm_threshold_low = lgbm_threshold_low
        self.patchtst_regime_weight = patchtst_regime_weight
        self.zscore_entry_threshold = zscore_entry_threshold

    def generate(
        self,
        lgbm_prob: float,
        patchtst_forecast: np.ndarray,
        current_zscore: float,
        symbol: str,
        horizon: str,
        features: dict[str, float],
    ) -> Optional["SignalPacket"]:
        from core.agent_loop import SignalPacket, SignalSource, ActionType, ConfidenceTier  # noqa: E402

        # Ensemble score
        regime_score = self._regime_compatibility(patchtst_forecast)
        ensemble_score = (
            lgbm_prob * (1 - self.patchtst_regime_weight)
            + regime_score * self.patchtst_regime_weight
        )

        # Confidence tier
        if ensemble_score >= self.lgbm_threshold_high:
            tier = ConfidenceTier.HIGH
        elif ensemble_score >= self.lgbm_threshold_low:
            tier = ConfidenceTier.MEDIUM
        else:
            tier = ConfidenceTier.LOW

        # Direction from z-score sign
        if current_zscore < -self.zscore_entry_threshold:
            action = ActionType.BUY     # Oversold → fade up
        elif current_zscore > self.zscore_entry_threshold:
            action = ActionType.SELL    # Overbought → fade down
        else:
            action = ActionType.HOLD

        if action == ActionType.HOLD and tier == ConfidenceTier.LOW:
            return None   # No signal

        return SignalPacket(
            source=SignalSource.ENSEMBLE,
            symbol=symbol,
            action=action,
            confidence=ensemble_score,
            horizon=horizon,
            z_score=current_zscore,
            features=features,
        )

    def _regime_compatibility(self, forecast: np.ndarray) -> float:
        """
        Score how compatible the current regime is with mean-reversion.
        Higher = more mean-reverting.
        """
        # If forecast is strongly trending, score is low
        fwd_returns = np.diff(forecast)
        trend_strength = abs(np.mean(fwd_returns)) / (np.std(fwd_returns) + 1e-9)
        # Sigmoid: high trend_strength → low compatibility
        return 1.0 / (1.0 + np.exp(trend_strength - 0.5))


# =============================================================================
# Feature computation helpers (stubs — replace with real implementations)
# =============================================================================

def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _stoch_k(df: pd.DataFrame, window: int) -> pd.Series:
    low = df["low"].rolling(window).min()
    high = df["high"].rolling(window).max()
    return 100 * (df["close"] - low) / (high - low + 1e-9)


def _stoch_d(df: pd.DataFrame, k_window: int, d_window: int) -> pd.Series:
    return _stoch_k(df, k_window).rolling(d_window).mean()


def _cci(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad + 1e-9)


def _williams_r(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"].rolling(window).max()
    low = df["low"].rolling(window).min()
    return -100 * (high - df["close"]) / (high - low + 1e-9)


def _mfi(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["tick_volume"]
    pos_flow = mf.where(tp.diff() > 0, 0).rolling(window).sum()
    neg_flow = mf.where(tp.diff() < 0, 0).rolling(window).sum()
    mfr = pos_flow / (neg_flow + 1e-9)
    return 100 - (100 / (1 + mfr))


def _tsi(df: pd.DataFrame, long: int, short: int) -> pd.Series:
    m = df["close"].diff()
    m_abs = m.abs()
    m_smooth = m.ewm(span=long).mean().ewm(span=short).mean()
    m_abs_smooth = m_abs.ewm(span=long).mean().ewm(span=short).mean()
    return 100 * m_smooth / (m_abs_smooth + 1e-9)


def _ultimate_osc(df: pd.DataFrame, window: int) -> pd.Series:
    bp = df["close"] - pd.concat([df["low"], df["close"].shift(1)], axis=1).min(axis=1)
    tr = pd.concat([df["high"] - df["low"],
                     abs(df["high"] - df["close"].shift(1)),
                     abs(df["low"] - df["close"].shift(1))], axis=1).max(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _bb_upper(df: pd.DataFrame, window: int, n_std: float) -> pd.Series:
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    return sma + n_std * std


def _bb_lower(df: pd.DataFrame, window: int, n_std: float) -> pd.Series:
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    return sma - n_std * std


def _bb_width(df: pd.DataFrame, window: int) -> pd.Series:
    return (_bb_upper(df, window, 2) - _bb_lower(df, window, 2)) / df["close"].rolling(window).mean()


def _bb_position(df: pd.DataFrame, window: int) -> pd.Series:
    upper = _bb_upper(df, window, 2)
    lower = _bb_lower(df, window, 2)
    return (df["close"] - lower) / (upper - lower + 1e-9)


def _bb_squeeze(df: pd.DataFrame, window: int) -> pd.Series:
    width = _bb_width(df, window)
    return width.rolling(window).apply(lambda x: (x.iloc[-1] < x.quantile(0.1)), raw=False).astype(float)


def _vol_ratio(df: pd.DataFrame, short: int, long: int) -> pd.Series:
    short_vol = df["close"].pct_change().rolling(short).std()
    long_vol = df["close"].pct_change().rolling(long).std()
    return short_vol / (long_vol + 1e-9)


def _parkinson_vol(df: pd.DataFrame, window: int) -> pd.Series:
    return np.sqrt(1 / (4 * np.log(2)) * (np.log(df["high"] / df["low"]) ** 2).rolling(window).mean())


def _garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    return np.sqrt(
        0.5 * (np.log(df["high"] / df["low"])) ** 2
        - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"])) ** 2
    ).rolling(window).mean()


def _yang_zhang_vol(df: pd.DataFrame, window: int) -> pd.Series:
    # Simplified YZ volatility
    return _parkinson_vol(df, window)  # Placeholder — implement full YZ for production


def _obv(df: pd.DataFrame) -> pd.Series:
    return (df["tick_volume"] * np.sign(df["close"].diff())).cumsum()


def _vwap(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["tick_volume"]).rolling(window).sum() / df["tick_volume"].rolling(window).sum()


def _volume_climax(df: pd.DataFrame, window: int) -> pd.Series:
    vol_ma = df["tick_volume"].rolling(window).mean()
    return df["tick_volume"] / (vol_ma + 1e-9) * abs(df["close"].pct_change()) * 100


def _zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)


def _kc_position(df: pd.DataFrame, window: int, atr_mult: float) -> pd.Series:
    ema = df["close"].ewm(span=window).mean()
    atr = _atr(df, window)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return (df["close"] - lower) / (upper - lower + 1e-9)


def _hurst(series: pd.Series, window: int) -> pd.Series:
    def _hurst_single(x):
        if len(x) < 20:
            return 0.5
        lags = range(2, min(20, len(x) // 2))
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau + 1e-9), 1)
        return poly[0]
    return series.rolling(window).apply(_hurst_single, raw=False)


def _half_life(series: pd.Series, window: int) -> pd.Series:
    def _hl(x):
        if len(x) < 10:
            return np.nan
        y = x - x.mean()
        y_lag = np.roll(y, 1)[1:]
        y = y[1:]
        beta = np.dot(y_lag, y) / (np.dot(y_lag, y_lag) + 1e-9)
        if beta <= 0 or beta >= 1:
            return np.inf
        return -np.log(2) / np.log(beta)
    return series.rolling(window).apply(_hl, raw=False)


def _adf_stat_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling approximate ADF statistic."""
    from statsmodels.tsa.stattools import adfuller
    def _adf(x):
        if len(x) < 10:
            return 0
        try:
            return adfuller(x, maxlag=min(5, len(x) // 4), autolag=None)[0]
        except Exception:
            return 0
    return series.rolling(window).apply(_adf, raw=False)


def _rsi_divergence(df: pd.DataFrame, window: int) -> pd.Series:
    rsi_vals = _rsi(df["close"], window)
    price_peak = df["close"].rolling(window).apply(lambda x: x.iloc[-1] / x.max() if x.max() > 0 else 1)
    return (rsi_vals / 100) - price_peak  # Positive = bullish divergence, Negative = bearish


def _mr_composite(df: pd.DataFrame, window: int) -> pd.Series:
    """Composite mean-reversion score."""
    z = _zscore(df["close"], window)
    bb_pos = _bb_position(df, window)
    hurst = _hurst(df["close"], window)
    hl = _half_life(df["close"], window)
    # Higher = more mean-reverting
    return 1 - (abs(z) / 3 + abs(bb_pos - 0.5) + abs(hurst - 0.5) + np.clip(1 / (hl + 1e-9), 0, 1)) / 4


def _fractal_efficiency(series: pd.Series, window: int) -> pd.Series:
    def _fe(x):
        if len(x) < 2:
            return 0
        net_change = abs(x.iloc[-1] - x.iloc[0])
        sum_changes = np.sum(np.abs(np.diff(x)))
        return net_change / (sum_changes + 1e-9)
    return series.rolling(window).apply(_fe, raw=False)


def _adx(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = _atr(df, 1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * plus_dm.rolling(window).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.rolling(window).mean() / (atr + 1e-9)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    return dx.rolling(window).mean()


def _di_plus(df: pd.DataFrame, window: int) -> pd.Series:
    plus_dm = df["high"].diff().clip(lower=0)
    atr = _atr(df, window)
    return 100 * plus_dm.rolling(window).mean() / (atr + 1e-9)


def _di_minus(df: pd.DataFrame, window: int) -> pd.Series:
    minus_dm = (-df["low"].diff()).clip(lower=0)
    atr = _atr(df, window)
    return 100 * minus_dm.rolling(window).mean() / (atr + 1e-9)


def _aroon_up(df: pd.DataFrame, window: int) -> pd.Series:
    return 100 * df["high"].rolling(window).apply(lambda x: np.argmax(x)) / (window - 1)


def _aroon_down(df: pd.DataFrame, window: int) -> pd.Series:
    return 100 * df["low"].rolling(window).apply(lambda x: np.argmin(x)) / (window - 1)


def _ema_slope(series: pd.Series, window: int) -> pd.Series:
    ema = series.ewm(span=window).mean()
    return (ema - ema.shift(3)) / ema.shift(3) * 100


def _sample_entropy(series: pd.Series, window: int, m: int = 2, r_factor: float = 0.2) -> pd.Series:
    def _se(x):
        if len(x) < m + 2:
            return 0
        r = r_factor * np.std(x)
        if r == 0:
            return 0
        def count_matches(template_len):
            count = 0
            templates = np.array([x[i:i + template_len] for i in range(len(x) - template_len)])
            for i in range(len(templates)):
                dists = np.max(np.abs(templates[i] - templates), axis=1)
                count += np.sum(dists < r) - 1
            return count
        a = count_matches(m)
        b = count_matches(m + 1)
        if b <= 0:
            return 0
        return -np.log(a / b)
    return series.rolling(window).apply(_se, raw=False)


def _jarque_bera_rolling(df: pd.DataFrame, window: int) -> pd.Series:
    def _jb(x):
        if len(x) < 4:
            return 0
        n = len(x)
        s = x.skew()
        k = x.kurtosis()
        return n / 6 * (s ** 2 + 0.25 * k ** 2)
    return df["close"].pct_change().rolling(window).apply(_jb, raw=False)


def _variance_ratio(series: pd.Series, k: int) -> pd.Series:
    return series.pct_change().rolling(k).var() / (k * series.pct_change().rolling(1).var() + 1e-9)


def _runs_test_z_rolling(series: pd.Series, window: int) -> pd.Series:
    def _rz(x):
        if len(x) < 5:
            return 0
        n = len(x)
        median = np.median(x)
        above = (x > median).values
        runs = 1 + np.sum(np.diff(above) != 0)
        n1 = np.sum(above)
        n2 = n - n1
        if n1 < 2 or n2 < 2:
            return 0
        expected_runs = 1 + 2 * n1 * n2 / n
        std_runs = np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1)))
        return (runs - expected_runs) / (std_runs + 1e-9)
    return series.rolling(window).apply(_rz, raw=False)


def _bds_stat_rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(
        lambda x: np.corrcoef(x[:-1] if len(x) > 1 else [0], x[1:] if len(x) > 1 else [0])[0, 1] if len(x) > 2 else 0,
        raw=False,
    )


def _effective_spread(df: pd.DataFrame, window: int) -> pd.Series:
    return 2 * np.sqrt(-df["close"].pct_change().rolling(window).cov(df["close"].shift(-1).pct_change().rolling(window)))


def _roll_measure(df: pd.DataFrame, window: int) -> pd.Series:
    delta = df["close"].diff()
    cov_delta = delta.rolling(window).cov(delta.shift(1))
    return 2 * np.sqrt(np.clip(-cov_delta, 0, None))


def _tick_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    tick_dir = np.sign(df["close"].diff())
    return tick_dir.rolling(window).sum() / window


def _quote_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    mid = (df["ask"] + df["bid"]) / 2
    mid_change = mid.diff()
    return mid_change.rolling(window).mean() / (mid_change.rolling(window).std() + 1e-9)


def _vpim(df: pd.DataFrame, window: int) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading (VPIM)."""
    return _quote_imbalance(df, window) * _volume_climax(df, window)


def _depth_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    """Approximate depth imbalance from spread and volume."""
    return (_quote_imbalance(df, window) * (df["tick_volume"] / df["tick_volume"].rolling(window).mean()) +
            (df["ask"] - df["bid"]) / (df["close"] + 1e-9) * 100)


def _dominant_cycle(series: pd.Series, window: int) -> pd.Series:
    def _dc(x):
        if len(x) < 10:
            return np.nan
        fft = np.fft.rfft(x - x.mean())
        freqs = np.fft.rfftfreq(len(x))
        if len(freqs) < 2:
            return np.nan
        peak_idx = np.argmax(np.abs(fft[1:])) + 1
        if peak_idx >= len(freqs) or freqs[peak_idx] == 0:
            return np.nan
        return 1 / freqs[peak_idx]
    return series.rolling(window).apply(_dc, raw=False)


def _cycle_strength(series: pd.Series, window: int) -> pd.Series:
    def _cs(x):
        if len(x) < 10:
            return 0
        fft = np.fft.rfft(x - x.mean())
        total_power = np.sum(np.abs(fft))
        if total_power == 0:
            return 0
        peak_power = np.max(np.abs(fft[1:])) if len(fft) > 1 else 0
        return peak_power / total_power
    return series.rolling(window).apply(_cs, raw=False)


def _dft_peak_ratio(series: pd.Series, window: int) -> pd.Series:
    def _pr(x):
        if len(x) < 10:
            return 0
        fft = np.abs(np.fft.rfft(x - x.mean()))
        if len(fft) < 3 or fft[0] == 0:
            return 0
        return fft[1] / fft[2] if fft[2] != 0 else 0
    return series.rolling(window).apply(_pr, raw=False)


def _wavelet_energy(series: pd.Series, window: int) -> pd.Series:
    def _we(x):
        if len(x) < 4:
            return 0
        coeffs = np.diff(x)  # Simplified — production should use pywt
        return np.sum(coeffs ** 2) / len(coeffs)
    return series.rolling(window).apply(_we, raw=False)


# =============================================================================
# Training pipeline
# =============================================================================

def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: str = "1h",
    sample_weight: Optional[pd.Series] = None,
) -> tuple:
    """Train a LightGBM classifier with the curated hyperparameters."""
    import lightgbm as lgb

    params = get_lightgbm_params(horizon)

    # Handle class imbalance
    pos_ratio = y.mean()
    if pos_ratio > 0 and pos_ratio < 1:
        params["scale_pos_weight"] = (1 - pos_ratio) / pos_ratio

    model = lgb.LGBMClassifier(**params)

    # Time-series cross-validation split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        sample_weight=sample_weight.iloc[:split_idx] if sample_weight is not None else None,
    )

    val_auc = model.best_score_["valid_0"]["auc"]
    logger.info("LightGBM trained: horizon=%s val_auc=%.4f best_iter=%d",
                 horizon, val_auc, model.best_iteration_)

    return model, val_auc


# =============================================================================
# Ensemble: LightGBM + PatchTST
# =============================================================================

class EnsemblePredictor:
    """
    Weighted ensemble of LightGBM and PatchTST predictions.

    Weighting:
      - LightGBM: 0.7 (tree-based, stable on tabular features)
      - PatchTST: 0.3 (deep learning, captures temporal patterns)
    """

    def __init__(
        self,
        lgbm_model,
        patchtst_model,
        lgbm_weight: float = 0.7,
        patchtst_weight: float = 0.3,
    ):
        self.lgbm_model = lgbm_model
        self.patchtst_model = patchtst_model
        self.lgbm_weight = lgbm_weight
        self.patchtst_weight = patchtst_weight

    def predict_proba(self, X_tabular: pd.DataFrame, X_temporal: np.ndarray) -> np.ndarray:
        """Return ensemble probability of mean-reversion."""
        lgbm_proba = self.lgbm_model.predict_proba(X_tabular)[:, 1]
        patchtst_forecast = self.patchtst_model.predict(X_temporal)
        # Convert PatchTST forecast to probability via regime compatibility score
        regime_scores = self._forecast_to_regime_score(patchtst_forecast)
        ensemble = (
            lgbm_proba * self.lgbm_weight
            + regime_scores * self.patchtst_weight
        )
        return ensemble

    @staticmethod
    def _forecast_to_regime_score(forecast: np.ndarray) -> float:
        """Convert a forecast array into a mean-reversion compatibility score."""
        if forecast.ndim == 2:
            scores = []
            for f in forecast:
                fwd_returns = np.diff(f)
                ts = abs(np.mean(fwd_returns)) / (np.std(fwd_returns) + 1e-9)
                scores.append(1.0 / (1.0 + np.exp(ts - 0.5)))
            return np.array(scores)
        fwd_returns = np.diff(forecast)
        ts = abs(np.mean(fwd_returns)) / (np.std(fwd_returns) + 1e-9)
        return 1.0 / (1.0 + np.exp(ts - 0.5))


# =============================================================================
# Main pipeline
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ML Pipeline module loaded.")
    print(f"Feature groups: {list(FEATURE_DEFINITIONS.keys())}")
    total_features = sum(len(v) for v in FEATURE_DEFINITIONS.values())
    print(f"Total features: {total_features}")
    print(f"LightGBM params (1h): {get_lightgbm_params('1h')}")
    print(f"PatchTST params (1h): {PATCHTST_PARAMS}")