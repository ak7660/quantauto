"""end-to-end automated workflow entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from quantauto.backtesting import BacktestResult, run_backtest
from quantauto.backtesting.cross_section import (
    forward_returns_wide,
    predictions_to_topk_long_short_weights,
    run_cross_sectional_from_wide,
)
from quantauto.data import (
    LoadedMarketData,
    MultiAssetMarketData,
    align_multi_asset_to_common_index,
    load_market_data,
    validate_loaded_market_data,
    validate_multi_asset_market_data,
)
from quantauto.features import build_preset_features
from quantauto.labels import (
    LabelSpec,
    build_label,
    build_label_set,
    make_direction_label,
    make_forward_return_label,
)
from quantauto.models.dataset import TrainingDataset, make_training_dataset
from quantauto.models.config import (
    DEFAULT_TRAINING_CONFIGS,
    TrainingConfig,
    parse_time_budget_minutes,
)
from quantauto.models.panel_dataset import build_panel_training_dataset
from quantauto.models.panel_trainer import train_panel_models
from quantauto.models.trainer import train_models
from quantauto.models.types import TrainerResult
from quantauto.workflows.combined import build_combined_multi_backtest
from quantauto.workflows.config import DEFAULT_PRESET, DEFAULT_TARGET_HORIZON
from quantauto.workflows.helpers import elaborate_backtest_results, elaborate_ml_results


@dataclass
class PipelineResults:
    loaded: LoadedMarketData
    features: pd.DataFrame
    label: pd.Series
    dataset: TrainingDataset
    trainer_result: TrainerResult
    backtest_result: BacktestResult
    ml_metrics: Mapping[str, Any]
    backtest_results: Mapping[str, Any]
    backtest_results_by_scope: Mapping[str, BacktestResult]


@dataclass
class RankingPanelResults:
    """outputs from `target_type=\"ranking\"` cross-sectional run."""

    multi_loaded: MultiAssetMarketData
    panel_trainer: TrainerResult
    cross_sectional_backtest: BacktestResult
    ml_metrics: Mapping[str, Any]
    backtest_results: Mapping[str, Any]
    test_predictions: pd.Series


@dataclass
class MultiPipelineResults:
    multi_loaded: MultiAssetMarketData
    by_symbol: Dict[str, PipelineResults]
    aggregate_summary: pd.DataFrame
    errors: Mapping[str, str]
    combined_backtest: Optional[BacktestResult] = None
    combined_backtest_results: Optional[Mapping[str, Any]] = None
    combined_weighting: Optional[str] = None
    ranking_panel: Optional[RankingPanelResults] = None


def _coerce_loaded(data: Union[pd.DataFrame, str, Path, LoadedMarketData]) -> LoadedMarketData:
    if isinstance(data, LoadedMarketData):
        return data
    return load_market_data(data, verbose=False)


def _normalize_verbose_level(verbose: Union[bool, int]) -> int:
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    try:
        v = int(verbose)
    except Exception as exc:  # pragma: no cover
        raise ValueError("verbose must be 0, 1, or 2") from exc
    if v not in (0, 1, 2):
        raise ValueError(f"verbose must be 0, 1, or 2, got {verbose!r}")
    return v


def _date_window_str(index: pd.Index) -> str:
    if len(index) == 0:
        return "empty"
    if isinstance(index, pd.MultiIndex):
        t = pd.DatetimeIndex(index.get_level_values(0))
    else:
        t = pd.DatetimeIndex(index)
    t = t.sort_values()
    return f"{t[0]} -> {t[-1]} (bars={len(t)})"


def _run_auto_multi_ranking(
    data: MultiAssetMarketData,
    *,
    target_horizon: int,
    feature_preset: str,
    test_split: float,
    walk_forward_folds: int,
    training_time_budget: Union[str, int, float],
    model_ids: Optional[Sequence[str]],
    purge_bars: int,
    embargo_bars: int,
    execution_shift: int,
    fee_bps: float,
    slippage_bps: float,
    enable_backtest_plots: bool,
    backtest_default_plots: Sequence[str],
    backtest_optional_plots: Sequence[str],
    ranking_top_k: int,
    min_common_bars: int,
    verbose: int,
) -> MultiPipelineResults:
    def _v(msg: str) -> None:
        if verbose >= 1:
            print(f"[run_auto:ranking] {msg}")

    _v("starting ranking multi-asset pipeline")
    n_sym = len(data.symbols)
    if 2 * int(ranking_top_k) > n_sym:
        raise ValueError(
            f"need at least 2*ranking_top_k={2 * ranking_top_k} symbols, got {n_sym}"
        )
    al = align_multi_asset_to_common_index(
        data,
        min_common_bars=max(int(min_common_bars), 5),
    )
    multi = al.data
    _v(f"aligned symbols={len(multi.symbols)} common_index_bars={len(next(iter(multi.by_symbol.values())).data)}")
    panel = build_panel_training_dataset(
        multi,
        feature_preset=feature_preset,  # type: ignore[arg-type]
        target_horizon=target_horizon,
        execution_shift=execution_shift,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[run_auto:ranking] panel dataset rows={len(panel.X)} cols={panel.X.shape[1]} time_points={len(panel.time_index)}"
        )
    base = DEFAULT_TRAINING_CONFIGS["ranking"]
    cfg = TrainingConfig(
        task_type="ranking",
        model_ids=base.model_ids,
        primary_metric=base.primary_metric,
        maximize_metric=base.maximize_metric,
        test_split=test_split,
        walk_forward_folds=walk_forward_folds,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        training_time_budget_minutes=parse_time_budget_minutes(training_time_budget),
        enable_layer2=False,
    )
    tr = train_panel_models(
        panel,
        task_type="ranking",
        model_ids=model_ids,
        config=cfg,
    )
    _v(f"training done best_model={tr.best_model_id}")
    if verbose >= 2:
        for mid, mr in tr.all_results.items():
            fm = mr.fitted_model
            p = getattr(fm, "params", None)
            print(f"[run_auto:ranking] model={mid} params={dict(p) if p is not None else {}}")
            for sr in mr.split_results:
                tw = _date_window_str(sr.actuals.index)
                print(
                    f"[run_auto:ranking] split={sr.split_id} train_size={sr.train_size} test_window={tw}"
                )
    br = tr.best_result
    split_res = br.split_results
    pred = pd.concat([s.predictions for s in split_res], axis=0)
    pred = pred[~pred.index.duplicated(keep="last")].sort_index()
    w_long = predictions_to_topk_long_short_weights(pred, k=ranking_top_k)
    w_wide = w_long.unstack(level=1, fill_value=0.0)
    w_wide = w_wide.reindex(columns=list(multi.symbols), fill_value=0.0)
    closes = {s: multi.by_symbol[s].data["close"] for s in multi.symbols}
    ret_w = forward_returns_wide(closes, list(multi.symbols), target_horizon)
    idx = w_wide.index.intersection(ret_w.index)
    w_wide = w_wide.reindex(idx).reindex(columns=ret_w.columns, fill_value=0.0)
    ret_w = ret_w.reindex(idx)
    cs = run_cross_sectional_from_wide(
        w_wide,
        ret_w,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        enable_plots=enable_backtest_plots,
        default_plots=tuple(backtest_default_plots),
        optional_plots=tuple(backtest_optional_plots),
        strategy_label=f"ranking_topk_k{ranking_top_k}_all_symbols",
    )
    _v(f"backtest done total_return={cs.summary.get('total_return', 0.0):.6f}")
    _v(f"backtest window: {_date_window_str(cs.returns.index)}")
    ml = elaborate_ml_results(tr)
    back = elaborate_backtest_results(cs)
    rp = RankingPanelResults(
        multi_loaded=multi,
        panel_trainer=tr,
        cross_sectional_backtest=cs,
        ml_metrics=ml,
        backtest_results=back,
        test_predictions=pred,
    )
    agg = pd.DataFrame(
        [dict(**cs.summary, symbol="CROSS_SECTION", best_model_id=tr.best_model_id)]
    ).set_index("symbol")
    return MultiPipelineResults(
        multi_loaded=multi,
        by_symbol={},
        aggregate_summary=agg,
        errors={},
        ranking_panel=rp,
    )


def _run_auto_single_loaded(
    loaded: LoadedMarketData,
    *,
    target_type: str = "regression",
    target_horizon: int = DEFAULT_TARGET_HORIZON,
    feature_preset: str = DEFAULT_PRESET,
    test_split: float = 0.2,
    walk_forward_folds: int = 1,
    training_time_budget: Union[str, int, float] = "30m",
    model_ids: Optional[Sequence[str]] = None,
    enable_layer2: bool = True,
    label_spec: Optional[LabelSpec] = None,
    label_specs: Optional[Sequence[LabelSpec]] = None,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    execution_shift: int = 1,
    threshold: float = 0.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    backtest_scope: str = "all_test_folds",
    enable_backtest_plots: bool = True,
    backtest_default_plots: Sequence[str] = ("equity_curve", "returns_distribution"),
    backtest_optional_plots: Sequence[str] = (),
    use_numba_backtest: bool = True,
    verbose: int = 1,
    asset_label: Optional[str] = None,
) -> PipelineResults:
    """run one complete single-asset automated workflow."""
    def _v(msg: str) -> None:
        if verbose >= 1:
            print(f"[run_auto:single] {msg}")

    _v(f"validating input rows={len(loaded.data)}")
    vr = validate_loaded_market_data(loaded)
    if not vr.ok:
        raise ValueError(f"input market data failed validation: {vr.errors}")

    engineered = build_preset_features(
        loaded,
        preset=feature_preset,  # type: ignore[arg-type]
        execution_shift=execution_shift,
        drop_warmup=False,
    )
    _v(f"features engineered rows={len(engineered.data)} cols={engineered.data.shape[1]}")
    if verbose >= 2:
        cols = list(engineered.data.columns)
        preview = ", ".join(cols[:50])
        if len(cols) > 50:
            preview += ", ..."
        _v(f"feature names: {preview}")

    close = loaded.data["close"]
    if label_specs is not None and len(label_specs) > 0:
        labels, _ = build_label_set(close, label_specs)
        first = label_specs[0]
        label = labels[first.name]
        model_task = (
            "classification" if first.target_type == "classification" else "regression"
        )
        target_horizon = first.horizon
    elif label_spec is not None:
        label, _ = build_label(close, label_spec)
        model_task = (
            "classification" if label_spec.target_type == "classification" else "regression"
        )
        target_horizon = label_spec.horizon
    elif target_type == "regression":
        label, _ = make_forward_return_label(
            close, horizon=target_horizon, name="target_next_return"
        )
        model_task = "regression"
    elif target_type == "classification":
        label, _ = make_direction_label(
            close,
            horizon=target_horizon,
            threshold=threshold,
            name="target_direction",
        )
        model_task = "classification"
    else:
        raise ValueError(
            f"unsupported target_type {target_type!r}; use regression or classification"
        )
    _v(f"label built task={model_task} horizon={target_horizon} label_name={label.name}")
    strategy_label_base = (
        f"{model_task}_{asset_label}" if asset_label is not None else f"{model_task}"
    )

    dataset = make_training_dataset(engineered, label, horizon=target_horizon)
    _v(f"dataset built rows={len(dataset.X)} cols={dataset.X.shape[1]}")
    base_cfg = DEFAULT_TRAINING_CONFIGS[model_task]
    cfg = TrainingConfig(
        task_type=base_cfg.task_type,
        model_ids=base_cfg.model_ids,
        primary_metric=base_cfg.primary_metric,
        maximize_metric=base_cfg.maximize_metric,
        test_split=test_split,
        walk_forward_folds=walk_forward_folds,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        training_time_budget_minutes=parse_time_budget_minutes(training_time_budget),
        enable_layer2=enable_layer2,
    )
    trainer_result = train_models(
        dataset,
        task_type=model_task,
        model_ids=model_ids,
        config=cfg,
    )
    _v(f"training done best_model={trainer_result.best_model_id}")
    if verbose >= 2:
        for mid, mr in trainer_result.all_results.items():
            fm = mr.fitted_model
            p = getattr(fm, "params", None)
            _v(f"model={mid} params={dict(p) if p is not None else {}}")
            for sr in mr.split_results:
                test_w = _date_window_str(sr.actuals.index)
                _v(f"split={sr.split_id} train_size={sr.train_size} test_window={test_w}")

    split_results = trainer_result.best_result.split_results
    if not split_results:
        raise RuntimeError("best model has no split results")
    last_pred = split_results[-1].predictions
    last_actual = split_results[-1].actuals

    pred_all = pd.concat([s.predictions for s in split_results], axis=0)
    act_all = pd.concat([s.actuals for s in split_results], axis=0)
    pred_all = pred_all[~pred_all.index.duplicated(keep="last")].sort_index()
    act_all = act_all[~act_all.index.duplicated(keep="last")].sort_index()

    market_forward_returns = close.pct_change(target_horizon).shift(-target_horizon)

    def _to_backtest_returns(y: pd.Series) -> pd.Series:
        if model_task == "classification":
            mapped = market_forward_returns.reindex(y.index).rename("forward_return")
            return mapped
        return y

    by_scope: dict[str, BacktestResult] = {}
    if backtest_scope in ("last_fold", "both"):
        by_scope["last_fold"] = run_backtest(
            last_pred,
            _to_backtest_returns(last_actual),
            task_type=model_task,
            threshold=threshold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            enable_plots=enable_backtest_plots,
            default_plots=tuple(backtest_default_plots),
            optional_plots=tuple(backtest_optional_plots),
            use_numba=use_numba_backtest,
            scope="last_fold",
            strategy_label=f"{strategy_label_base}_last_fold",
        )
    if backtest_scope in ("all_test_folds", "both"):
        by_scope["all_test_folds"] = run_backtest(
            pred_all,
            _to_backtest_returns(act_all),
            task_type=model_task,
            threshold=threshold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            enable_plots=enable_backtest_plots,
            default_plots=tuple(backtest_default_plots),
            optional_plots=tuple(backtest_optional_plots),
            use_numba=use_numba_backtest,
            scope="all_test_folds",
            strategy_label=f"{strategy_label_base}_all_test_folds",
        )
    if not by_scope:
        raise ValueError(
            f"unsupported backtest_scope {backtest_scope!r}; use last_fold, all_test_folds, or both"
        )
    bt = by_scope.get("all_test_folds") or by_scope.get("last_fold")
    assert bt is not None
    _v(f"backtest done scope={bt.scope} total_return={bt.summary.get('total_return', 0.0):.6f}")
    _v(f"backtest window: {_date_window_str(bt.returns.index)}")

    ml = elaborate_ml_results(trainer_result)
    backtest = elaborate_backtest_results(bt)
    return PipelineResults(
        loaded=loaded,
        features=engineered.data,
        label=label,
        dataset=dataset,
        trainer_result=trainer_result,
        backtest_result=bt,
        ml_metrics=ml,
        backtest_results=backtest,
        backtest_results_by_scope=by_scope,
    )


def run_auto(
    data: Union[pd.DataFrame, str, Path, LoadedMarketData, MultiAssetMarketData],
    *,
    target_type: str = "regression",
    target_horizon: int = DEFAULT_TARGET_HORIZON,
    feature_preset: str = DEFAULT_PRESET,
    test_split: float = 0.2,
    walk_forward_folds: int = 1,
    training_time_budget: Union[str, int, float] = "30m",
    model_ids: Optional[Sequence[str]] = None,
    enable_layer2: bool = True,
    label_spec: Optional[LabelSpec] = None,
    label_specs: Optional[Sequence[LabelSpec]] = None,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    execution_shift: int = 1,
    threshold: float = 0.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    backtest_scope: str = "all_test_folds",
    enable_backtest_plots: bool = True,
    backtest_default_plots: Sequence[str] = ("equity_curve", "returns_distribution"),
    backtest_optional_plots: Sequence[str] = (),
    use_numba_backtest: bool = True,
    continue_on_error: bool = False,
    multi_portfolio: str = "per_symbol_only",
    multi_weighting: str = "equal",
    inv_vol_lookback: int = 20,
    ranking_top_k: int = 1,
    min_common_bars: int = 5,
    verbose: int = 1,
) -> Union[PipelineResults, MultiPipelineResults]:
    """run automated pipeline for single or multi-asset input."""
    verbose = _normalize_verbose_level(verbose)
    if verbose:
        print("[run_auto] starting pipeline")
    if isinstance(data, MultiAssetMarketData):
        if verbose:
            print(f"[run_auto] multi-asset input symbols={len(data.symbols)}")
        v_multi = validate_multi_asset_market_data(data)
        if not v_multi.ok:
            raise ValueError(f"input multi-asset data failed validation: {v_multi.errors}")

        if target_type == "ranking":
            if label_spec is not None or (label_specs is not None and len(label_specs) > 0):
                raise ValueError("custom label_spec(s) are not supported for target_type=ranking")
            return _run_auto_multi_ranking(
                data,
                target_horizon=target_horizon,
                feature_preset=feature_preset,
                test_split=test_split,
                walk_forward_folds=walk_forward_folds,
                training_time_budget=training_time_budget,
                model_ids=model_ids,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
                execution_shift=execution_shift,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                enable_backtest_plots=enable_backtest_plots,
                backtest_default_plots=backtest_default_plots,
                backtest_optional_plots=backtest_optional_plots,
                ranking_top_k=ranking_top_k,
                min_common_bars=min_common_bars,
                verbose=verbose,
            )

        by_symbol: Dict[str, PipelineResults] = {}
        errors: Dict[str, str] = {}
        for sym, loaded in data.by_symbol.items():
            try:
                if verbose:
                    print(f"[run_auto] processing symbol={sym}")
                by_symbol[sym] = _run_auto_single_loaded(
                    loaded=loaded,
                    target_type=target_type,
                    target_horizon=target_horizon,
                    feature_preset=feature_preset,
                    test_split=test_split,
                    walk_forward_folds=walk_forward_folds,
                    training_time_budget=training_time_budget,
                    model_ids=model_ids,
                    enable_layer2=enable_layer2,
                    label_spec=label_spec,
                    label_specs=label_specs,
                    purge_bars=purge_bars,
                    embargo_bars=embargo_bars,
                    execution_shift=execution_shift,
                    threshold=threshold,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    backtest_scope=backtest_scope,
                    enable_backtest_plots=enable_backtest_plots,
                    backtest_default_plots=backtest_default_plots,
                    backtest_optional_plots=backtest_optional_plots,
                    use_numba_backtest=use_numba_backtest,
                    verbose=verbose,
                    asset_label=sym,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                errors[sym] = str(exc)
        if not by_symbol:
            raise RuntimeError("multi-asset run produced no successful symbol results")
        rows = []
        for sym, res in by_symbol.items():
            s = dict(res.backtest_result.summary)
            s["symbol"] = sym
            s["best_model_id"] = res.trainer_result.best_model_id
            rows.append(s)
        agg = pd.DataFrame(rows).set_index("symbol").sort_index()
        comb: Optional[BacktestResult] = None
        comb_results: Optional[Mapping[str, Any]] = None
        mw: Optional[str] = None
        if multi_portfolio in ("combined", "both") and by_symbol:
            comb = build_combined_multi_backtest(
                by_symbol,
                weighting=multi_weighting,  # type: ignore[arg-type]
                inv_vol_lookback=inv_vol_lookback,
                enable_plots=enable_backtest_plots,
                default_plots=tuple(backtest_default_plots),
                optional_plots=tuple(backtest_optional_plots),
            )
            comb_results = elaborate_backtest_results(comb)
            mw = multi_weighting
            c_row = {**dict(comb.summary), "symbol": "COMBINED", "best_model_id": "n/a"}
            agg = pd.concat(
                [agg, pd.DataFrame([c_row]).set_index("symbol")],
            ).sort_index()
        return MultiPipelineResults(
            multi_loaded=data,
            by_symbol=by_symbol,
            aggregate_summary=agg,
            errors=errors,
            combined_backtest=comb,
            combined_backtest_results=comb_results,
            combined_weighting=mw,
        )

    loaded = _coerce_loaded(data)
    if target_type == "ranking":
        raise ValueError("target_type='ranking' is only valid for MultiAssetMarketData")
    return _run_auto_single_loaded(
        loaded=loaded,
        target_type=target_type,
        target_horizon=target_horizon,
        feature_preset=feature_preset,
        test_split=test_split,
        walk_forward_folds=walk_forward_folds,
        training_time_budget=training_time_budget,
        model_ids=model_ids,
        enable_layer2=enable_layer2,
        label_spec=label_spec,
        label_specs=label_specs,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        execution_shift=execution_shift,
        threshold=threshold,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        backtest_scope=backtest_scope,
        enable_backtest_plots=enable_backtest_plots,
        backtest_default_plots=backtest_default_plots,
        backtest_optional_plots=backtest_optional_plots,
        use_numba_backtest=use_numba_backtest,
        verbose=verbose,
    )
