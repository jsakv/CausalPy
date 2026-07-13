#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for hierarchical difference in differences."""

from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from arviz import InferenceData

from causalpy.custom_exceptions import DataException, FormulaException
from causalpy.experiments import hierarchical_difference_in_differences as hdid_module
from causalpy.experiments.hierarchical_difference_in_differences import (
    HierarchicalDifferenceInDifferences,
)
from causalpy.pymc_models import HierarchicalLinearRegression, PyMCModel


class _FixedPosteriorHDiDModel(PyMCModel):
    """PyMCModel test double with deterministic hierarchical posterior draws."""

    def __init__(
        self,
        beta_fixed: np.ndarray | None = None,
        beta_random: np.ndarray | None = None,
        mu: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.beta_fixed = beta_fixed
        self.beta_random = beta_random
        self.mu = mu

    def fit(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.fit_kwargs = kwargs
        beta_fixed = self.beta_fixed
        if beta_fixed is None:
            beta_fixed = np.zeros((1, 2, 1, len(kwargs["coords"]["coeffs"])))
        beta_random = self.beta_random
        if beta_random is None:
            beta_random = np.zeros(
                (
                    1,
                    2,
                    len(kwargs["coords"]["groups"]),
                    1,
                    len(kwargs["coords"]["random_coeffs"]),
                )
            )
        mu = self.mu
        if mu is None:
            mu = np.zeros((1, 2, len(kwargs["coords"]["obs_ind"]), 1))
        sigma_fixed = np.ones((1, 2, 1))
        sigma_fixed[0, 1, 0] = 1.2
        sigma_random = np.ones((1, 2, 1, len(kwargs["coords"]["random_coeffs"])))
        sigma_random[0, 1, :, :] = 1.5
        posterior = xr.Dataset(
            data_vars={
                "beta_fixed": (
                    ["chain", "draw", "treated_units", "coeffs"],
                    beta_fixed,
                ),
                "sigma_fixed": (
                    ["chain", "draw", "treated_units"],
                    sigma_fixed,
                ),
                "sigma_random": (
                    ["chain", "draw", "treated_units", "random_coeffs"],
                    sigma_random,
                ),
                "beta_random": (
                    ["chain", "draw", "groups", "treated_units", "random_coeffs"],
                    beta_random,
                ),
                "mu": (
                    ["chain", "draw", "obs_ind", "treated_units"],
                    mu,
                ),
            },
            coords={
                "chain": [0],
                "draw": [0, 1],
                "treated_units": kwargs["coords"]["treated_units"],
                "coeffs": kwargs["coords"]["coeffs"],
                "random_coeffs": kwargs["coords"]["random_coeffs"],
                "groups": kwargs["coords"]["groups"],
                "obs_ind": kwargs["coords"]["obs_ind"],
            },
        )
        self.idata = InferenceData(posterior=posterior)


class TestHierarchicalDifferenceInDifferencesInterface:
    @staticmethod
    def _panel_data() -> pd.DataFrame:
        rows = []
        for store_idx in range(10):
            store_id = f"s{store_idx + 1}"
            treated = int(store_idx >= 5)
            for customer_idx in range(3):
                for month in [1, 2]:
                    rows.append(
                        {
                            "store_id": store_id,
                            "customer_id": f"{store_id}_c{customer_idx}",
                            "month": month,
                            "treated": treated,
                            "post_treatment": int(month == 2),
                            "y": float(10 + treated + month),
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _matrices(
        data: pd.DataFrame,
        *,
        include_did: bool = True,
        include_random_did: bool = True,
    ) -> SimpleNamespace:
        fixed = {
            "1": np.ones(len(data)),
            "post_treatment": data["post_treatment"].to_numpy(),
            "treated": data["treated"].to_numpy(),
        }
        if include_did:
            fixed["post_treatment:treated"] = (
                data["post_treatment"].to_numpy() * data["treated"].to_numpy()
            )
        rhs = pd.DataFrame(fixed, index=data.index)
        lhs = pd.DataFrame({"y": data["y"].to_numpy()}, index=data.index)
        random_effects = {"1|store_id": np.ones(len(data))}
        if include_random_did:
            random_effects["post_treatment:treated|store_id"] = (
                data["post_treatment"].to_numpy() * data["treated"].to_numpy()
            )
        z_matrix = pd.DataFrame(random_effects, index=data.index)
        group_idx, group_labels = pd.factorize(data["store_id"], sort=False)
        return SimpleNamespace(
            lhs=lhs,
            rhs=rhs,
            Z=z_matrix,
            metadata={
                "outcome_name": "y",
                "has_random_effects": True,
                "fixed_effect_names": list(rhs.columns),
                "random_effect_names": list(z_matrix.columns),
                "group": {
                    "variable": "store_id",
                    "labels": [str(label) for label in group_labels.tolist()],
                    "n_groups": int(len(group_labels)),
                    "idx": group_idx.astype(np.int32),
                    "components": [
                        "(post_treatment:treated | store_id)"
                        if include_random_did
                        else "(1 | store_id)"
                    ],
                },
            },
        )

    @staticmethod
    def _patch_parse_formula(
        monkeypatch: pytest.MonkeyPatch, matrices: SimpleNamespace
    ) -> None:
        monkeypatch.setattr(
            hdid_module, "parse_formula", lambda formula, data: matrices
        )

    @staticmethod
    def _experiment(
        data: pd.DataFrame,
        *,
        model: PyMCModel | None = None,
        non_centered: bool = True,
    ) -> HierarchicalDifferenceInDifferences:
        if model is None:
            model = _FixedPosteriorHDiDModel()
        return HierarchicalDifferenceInDifferences(
            data=data,
            formula=(
                "y ~ 1 + post_treatment + treated + post_treatment:treated "
                "+ (post_treatment:treated | store_id)"
            ),
            time_variable_name="month",
            unit_variable_name="customer_id",
            model=model,
            non_centered=non_centered,
        )

    def test_group_variable_is_inferred_from_random_effects_formula(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that the grouping variable is inferred from the random-effects formula."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)

        result = self._experiment(data)

        assert result.group_variable_name == "store_id"
        assert result.group_labels == [f"s{idx}" for idx in range(1, 11)]
        np.testing.assert_array_equal(
            result.group_idx, matrices.metadata["group"]["idx"]
        )

    def test_group_variable_is_inferred_from_real_parser(self) -> None:
        """Validate that the parser provides random-effects group metadata."""
        data = self._panel_data()

        result = self._experiment(data)

        assert result.group_variable_name == "store_id"
        assert result.group_labels == [f"s{idx}" for idx in range(1, 11)]
        assert result.n_groups == 10
        assert "post_treatment:treated|store_id" in result.random_effect_labels

    def test_formula_includes_random_effects(self) -> None:
        """Reject parsed DiD formulas without a random-effects term."""
        data = self._panel_data()

        with pytest.raises(FormulaException, match="requires a random-effects term"):
            HierarchicalDifferenceInDifferences(
                data=data,
                formula="y ~ 1 + post_treatment * treated",
                time_variable_name="month",
                unit_variable_name="customer_id",
                model=_FixedPosteriorHDiDModel(),
            )

    def test_formula_includes_did_interaction(self) -> None:
        """Reject parsed hierarchical formulas without the DiD interaction."""
        data = self._panel_data()

        with pytest.raises(FormulaException, match="exactly one DiD interaction"):
            HierarchicalDifferenceInDifferences(
                data=data,
                formula="y ~ 1 + post_treatment + treated + (1 | store_id)",
                time_variable_name="month",
                unit_variable_name="customer_id",
                model=_FixedPosteriorHDiDModel(),
            )

    def test_missing_group_metadata_raises_formula_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that missing parser group metadata is rejected."""
        data = self._panel_data()
        matrices = self._matrices(data)
        matrices.metadata["group"]["variable"] = None
        self._patch_parse_formula(monkeypatch, matrices)

        with pytest.raises(FormulaException, match="grouping variable"):
            self._experiment(data)

    def test_prepare_data_uses_mixed_model_matrices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that data preparation consumes mixed model matrices."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)

        result = self._experiment(data)

        assert result.outcome_variable_name == "y"
        assert result.labels == list(matrices.rhs.columns)
        assert result.random_effect_labels == list(matrices.Z.columns)
        assert result.coords["groups"] == [f"s{idx}" for idx in range(1, 11)]
        assert result.X.dims == ("obs_ind", "coeffs")
        assert result.Z.dims == ("obs_ind", "random_coeffs")
        assert result.y.dims == ("obs_ind", "treated_units")

    def test_att_selects_did_coefficient_posterior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that ATT is the posterior for the DiD fixed-effect coefficient."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)
        beta_fixed = np.zeros((1, 2, 1, len(matrices.metadata["fixed_effect_names"])))
        did_idx = matrices.metadata["fixed_effect_names"].index(
            "post_treatment:treated"
        )
        beta_fixed[:, :, :, did_idx] = np.array([[[2.5], [3.5]]])
        model = _FixedPosteriorHDiDModel(beta_fixed=beta_fixed)

        result = self._experiment(data, model=model)

        assert result.did_term == "post_treatment:treated"
        expected = result.idata.posterior["beta_fixed"].sel(coeffs=result.did_term)
        xr.testing.assert_identical(result.att, expected)
        xr.testing.assert_identical(result.causal_impact, expected)

    def test_group_effects_selects_random_effects_posterior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that group effects are the random-effects posterior."""
        data = self._panel_data()
        matrices = self._matrices(data)
        self._patch_parse_formula(monkeypatch, matrices)

        result = self._experiment(data)

        expected = result.idata.posterior["beta_random"]
        xr.testing.assert_identical(result.group_effects, expected)

    def test_get_plot_data_bayesian_returns_observed_panel_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that Bayesian plot data starts from the observed panel."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)

        plot_data = result.get_plot_data_bayesian()

        assert isinstance(plot_data, pd.DataFrame)
        assert plot_data.shape[0] == data.shape[0]
        assert {
            "store_id",
            "customer_id",
            "month",
            "treated",
            "post_treatment",
            "y",
            "y_fitted",
            "y_fitted_lower",
            "y_fitted_upper",
            "y_counterfactual",
            "y_counterfactual_lower",
            "y_counterfactual_upper",
        } <= set(plot_data.columns)

    def test_counterfactual_removes_fixed_and_random_did_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate counterfactual predictions remove fixed and group-level DiD terms."""
        data = self._panel_data()
        matrices = self._matrices(data, include_random_did=True)
        self._patch_parse_formula(monkeypatch, matrices)

        coeffs = matrices.metadata["fixed_effect_names"]
        random_coeffs = matrices.metadata["random_effect_names"]
        n_groups = matrices.metadata["group"]["n_groups"]
        n_obs = data.shape[0]
        beta_fixed = np.zeros((1, 2, 1, len(coeffs)))
        beta_fixed[0, :, 0, coeffs.index("post_treatment:treated")] = [5.0, 6.0]
        beta_random = np.zeros((1, 2, n_groups, 1, len(random_coeffs)))
        beta_random[
            0,
            :,
            :,
            0,
            random_coeffs.index("post_treatment:treated|store_id"),
        ] = np.array([2.0, 3.0])[:, None]
        mu = np.zeros((1, 2, n_obs, 1))
        mu[0, 0, :, 0] = 20.0
        mu[0, 1, :, 0] = 24.0
        model = _FixedPosteriorHDiDModel(
            beta_fixed=beta_fixed,
            beta_random=beta_random,
            mu=mu,
        )

        result = self._experiment(data, model=model)
        assert result.y_pred_counterfactual is not None
        plot_data = result.get_plot_data_bayesian()
        treated_post = plot_data["treated"].astype(bool) & plot_data[
            "post_treatment"
        ].astype(bool)

        np.testing.assert_allclose(
            plot_data.loc[treated_post, "y_counterfactual"],
            14.0,
        )
        np.testing.assert_allclose(
            plot_data.loc[~treated_post, "y_counterfactual"],
            22.0,
        )

    def test_public_plot_returns_figure_and_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that the public plot method renders Matplotlib objects."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)

        fig, ax = result.plot(show=False)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_bayesian_plot_returns_figure_and_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that the Bayesian plot method returns Matplotlib objects."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)

        fig, ax = result._bayesian_plot()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_variance_components_does_not_mutate_idata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that variance plotting does not add variables to the fit idata."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)
        posterior_vars = set(result.idata.posterior.data_vars)

        fig, ax = result.plot_variance_components(show=False)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert set(result.idata.posterior.data_vars) == posterior_vars
        plt.close(fig)

    def test_summary_prints_hierarchical_sections(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Validate that summary reports hierarchical model quantities."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        result = self._experiment(data)

        result.summary()

        captured = capsys.readouterr().out
        assert "Results:" in captured
        assert "Variance components:" in captured
        assert "Fixed effects:" in captured
        assert "Group-level ATT deviations:" in captured
        assert "Model coefficients:" not in captured

    def test_fit_model_delegates_to_hierarchical_regression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that fitting delegates to hierarchical regression inputs."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data))
        model = _FixedPosteriorHDiDModel()
        result = self._experiment(data, model=model, non_centered=False)

        assert model.fit_kwargs["X"] is result.X
        assert model.fit_kwargs["Z"] is result.Z
        assert model.fit_kwargs["y"] is result.y
        np.testing.assert_array_equal(model.fit_kwargs["group_idx"], result.group_idx)
        assert model.fit_kwargs["coords"] is result.coords
        assert model.fit_kwargs["non_centered"] is False

    @pytest.mark.integration
    def test_real_hierarchical_regression_backend_with_mocked_sampling(
        self, mock_pymc_sample: None
    ) -> None:
        """Validate HDiD integration with the hierarchical PyMC backend."""
        data = self._panel_data()
        model = HierarchicalLinearRegression(
            sample_kwargs={
                "draws": 2,
                "chains": 1,
                "progressbar": False,
                "random_seed": 42,
            }
        )

        result = HierarchicalDifferenceInDifferences(
            data=data,
            formula=(
                "y ~ 1 + post_treatment + treated + post_treatment:treated "
                "+ (post_treatment:treated | store_id)"
            ),
            time_variable_name="month",
            unit_variable_name="customer_id",
            model=model,
        )

        assert isinstance(result.model, HierarchicalLinearRegression)
        assert result.did_term == "post_treatment:treated"
        assert result.y_pred_counterfactual is not None
        assert result.causal_impact.dims == ("chain", "draw", "treated_units")
        assert {
            "beta_fixed",
            "beta_random",
            "sigma_fixed",
            "sigma_random",
            "mu",
        } <= set(result.idata.posterior.data_vars)

    def test_validation_rejects_unbalanced_panel_before_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that unbalanced panels are rejected before fitting."""
        data = self._panel_data().iloc[:-1].copy()
        self._patch_parse_formula(monkeypatch, self._matrices(data))

        with pytest.raises(DataException, match="balanced panel"):
            self._experiment(data)

    def test_formula_must_have_unambiguous_did_interaction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validate that the formula exposes exactly one DiD interaction term."""
        data = self._panel_data()
        self._patch_parse_formula(monkeypatch, self._matrices(data, include_did=False))

        with pytest.raises(FormulaException, match="exactly one DiD interaction"):
            HierarchicalDifferenceInDifferences(
                data=data,
                formula="y ~ 1 + post_treatment + treated + (1 | store_id)",
                time_variable_name="month",
                unit_variable_name="customer_id",
                model=_FixedPosteriorHDiDModel(),
            )
