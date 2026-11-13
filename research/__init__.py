# Исследовательский модуль DeepSeek Trading Architect
# Пополняемые базы: индикаторы, стратегии, позиционный менеджмент, риски, тестирование
# Реестры: идеи + стратегии

from indicators.indicator_registry import (
    INDICATOR_REGISTRY, IndicatorEnumerator, METHOD_MANIFEST,
    EXISTING_SIGNAL_TEMPLATES, EXISTING_FILTERS, EXISTING_EXITS,
    EXISTING_RISK_PROFILES, EXISTING_TESTING_METHODS, EXISTING_COST_COMPONENTS,
)
from costs.cost_engine import CostEngine, PIPFARM_COSTS, TICKMILL_COSTS, ACTIVE_COST_SET
from testing.null_models import NullModelEngine
from risk_mgmt.risk_profiles import (
    RiskProfileEngine, PIPFARM_PROFILES, FREE_PROFILE, CLASSIC_PROFILE, ALL_PROFILES, profile_gate_pass,
)
from catalogs.idea_strategy_registry import (
    UnifiedRegistry, IdeasRegistry, StrategyRegistry,
    IdeaEntry, StrategyEntry, IdeaStatus, SourceType, AssetClass,
    STATUS_TRANSITIONS, seed_ideas, seed_strategies,
)
from catalogs.catalogs import (
    STRATEGIES_CATALOG, POSITION_METHODS, RISK_METHODS, TESTING_METHODS, ALL_CATALOGS,
)