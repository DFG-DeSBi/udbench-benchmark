"""UCI/OpenML dataset loader.

Provides `UCILoader`, a class that fetches and pre-processes tabular regression
datasets from the UCI ML Repository and OpenML. Used by both:

- `data/generation/workflow.py` — to obtain training data for GP fitting.
- `data/runtime/dataset.py` — to build the feature pool for semi-synthetic
  presets at `DataSet.generate_dataset()` time.

Datasets are identified by positive UCI IDs, a small set of negative IDs for
built-in non-UCI sources (e.g. California Housing via scikit-learn), and a set
of direct OpenML data IDs for datasets not hosted on UCI.
"""

from __future__ import annotations

import logging
import time
from textwrap import shorten
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from ucimlrepo import fetch_ucirepo
from ucimlrepo.fetch import DatasetNotFoundError

logger = logging.getLogger(__name__)


CALIFORNIA_HOUSING_DATASET_ID = -1001

_SPECIAL_DATASET_SPECS: Dict[int, Dict[str, Any]] = {
    CALIFORNIA_HOUSING_DATASET_ID: {
        "name": "California Housing",
        "repository_url": (
            "https://scikit-learn.org/stable/modules/generated/"
            "sklearn.datasets.fetch_california_housing.html"
        ),
        "abstract": (
            "1990 California census block-group statistics used to predict median house "
            "value from 8 numeric socio-economic and geographic covariates."
        ),
        "target_column": "MedHouseVal",
        "feature_descriptions": {
            "MedInc": "Median income in block group",
            "HouseAge": "Median house age in block group",
            "AveRooms": "Average number of rooms per household",
            "AveBedrms": "Average number of bedrooms per household",
            "Population": "Block group population",
            "AveOccup": "Average household occupancy",
            "Latitude": "Block group latitude",
            "Longitude": "Block group longitude",
        },
        "target_description": "Median house value in units of $100,000",
        "loader_name": "fetch_california_housing",
    },
}

_DIRECT_OPENML_DATASET_SPECS: Dict[int, Dict[str, Any]] = {
    41021: {"name": "Moneyball"},
    44964: {"name": "Superconductivity"},
    44963: {"name": "Physicochemical Properties of Protein Tertiary Structure"},
    44969: {"name": "Naval Propulsion Plant"},
    44971: {"name": "White Wine Quality"},
    44972: {"name": "Red Wine Quality"},
    44976: {"name": "SARCOS"},
    44977: {"name": "California Housing"},
    44978: {
        "name": "CPU Activity",
        "target_column": "usr",
        "feature_descriptions": {
            "lread": "Reads (transfers per second) by the disk subsystem",
            "lwrite": "Writes (transfers per second) by the disk subsystem",
            "scall": "Number of system calls of all types per second",
            "sread": "Number of read system calls per second",
            "swrite": "Number of write system calls per second",
            "fork": "Number of fork system calls per second",
            "exec": "Number of exec system calls per second",
            "rchar": "Number of characters transferred by read system calls per second",
            "wchar": "Number of characters transferred by write system calls per second",
            "pgout": "Number of page out requests per second",
            "ppgout": "Number of pages paged out per second",
            "pgfree": "Number of pages placed on the free list per second",
            "pgscan": "Number of pages checked if they can be freed per second",
            "atch": "Number of page attaches (returning a reclaimed page) per second",
            "pgin": "Number of page in requests per second",
            "ppgin": "Number of pages paged in per second",
            "pflt": "Number of page faults caused by protection errors per second",
            "vflt": "Number of page faults caused by address translation per second",
            "runqsz": "Process run queue size (processes waiting for CPU)",
            "freemem": "Number of memory pages available to user processes",
            "freeswap": "Number of disk blocks available for page swapping",
        },
        "target_description": "Fraction of CPU time spent in user mode (0–100)",
    },
    44979: {"name": "Diamonds"},
    44980: {
        "name": "kin8nm",
        "target_column": "y",
        "feature_descriptions": {
            "x1": "Joint-angle input 1 of the 8-link robot arm",
            "x2": "Joint-angle input 2 of the 8-link robot arm",
            "x3": "Joint-angle input 3 of the 8-link robot arm",
            "x4": "Joint-angle input 4 of the 8-link robot arm",
            "x5": "Joint-angle input 5 of the 8-link robot arm",
            "x6": "Joint-angle input 6 of the 8-link robot arm",
            "x7": "Joint-angle input 7 of the 8-link robot arm",
            "x8": "Joint-angle input 8 of the 8-link robot arm",
        },
        "target_description": "Forward-kinematics output for the 8-link robot arm",
    },
    44981: {"name": "pumadyn32nh"},
    44983: {"name": "Miami Housing"},
    44984: {"name": "CPS 1988 Wages"},
    44987: {"name": "Socmob"},
    44989: {"name": "Kings County House Prices"},
    44990: {"name": "Brazilian Houses"},
    44992: {"name": "FPS Benchmark"},
    44993: {"name": "Health Insurance"},
    44994: {"name": "Cars"},
    45012: {"name": "FIFA"},
    45402: {"name": "Space GA"},
}

_OPENML_UCI_FALLBACKS: Dict[int, Dict[str, Any]] = {
    1: {
        "openml_data_id": 183,
        "name": "Abalone",
        "repository_url": "https://archive.ics.uci.edu/dataset/1/abalone",
        "abstract": (
            "Predict the age of abalone from physical measurements. Age is determined "
            "by counting shell rings under a microscope; the dataset provides physical "
            "measurements as a cheaper proxy."
        ),
        "target_column": "Class_Rings",
        "feature_descriptions": {
            "Sex": "M, F, and I (infant)",
            "Length": "Longest shell measurement (mm)",
            "Diameter": "Perpendicular to length (mm)",
            "Height": "With meat in shell (mm)",
            "Whole_weight": "Whole abalone weight (grams)",
            "Shucked_weight": "Weight of meat (grams)",
            "Viscera_weight": "Gut weight after bleeding (grams)",
            "Shell_weight": "Shell weight after drying (grams)",
        },
        "target_description": "Number of rings; add 1.5 for age in years",
    },
    165: {
        "openml_data_id": 4353,
        "name": "Concrete Compressive Strength",
        "repository_url": "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        "abstract": (
            "Concrete compressive strength is a highly nonlinear function of age and "
            "ingredients. Dataset contains 1030 instances with 8 quantitative input "
            "variables and one continuous output."
        ),
        "target_column": "concrete_compressive_strength",
        "feature_descriptions": {
            "cement": "kg per m³ mixture",
            "blast_furnace_slag": "kg per m³ mixture",
            "fly_ash": "kg per m³ mixture",
            "water": "kg per m³ mixture",
            "superplasticizer": "kg per m³ mixture",
            "coarse_aggregate": "kg per m³ mixture",
            "fine_aggregate": "kg per m³ mixture",
            "age": "days (1–365)",
        },
        "target_description": "Concrete compressive strength (MPa)",
    },
    186: {
        "openml_data_id": 40691,
        "name": "Wine Quality",
        "repository_url": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "abstract": (
            "Two datasets for red and white vinho verde wine samples from northern "
            "Portugal. Inputs are physicochemical tests; output is sensory quality score."
        ),
        "target_column": "class",
        "feature_descriptions": {
            "fixed_acidity": "fixed acidity",
            "volatile_acidity": "volatile acidity",
            "citric_acid": "citric acid",
            "residual_sugar": "residual sugar",
            "chlorides": "chlorides",
            "free_sulfur_dioxide": "free sulfur dioxide",
            "total_sulfur_dioxide": "total sulfur dioxide",
            "density": "density",
            "pH": "pH",
            "sulphates": "sulphates",
            "alcohol": "alcohol",
        },
        "target_description": "Wine quality score (0–10)",
    },
    242: {
        "openml_data_id": 573,
        "name": "Energy Efficiency",
        "repository_url": "https://archive.ics.uci.edu/dataset/242/energy+efficiency",
        "abstract": (
            "Energy analysis of 12 different building shapes simulated in Ecotect. "
            "Two targets: Heating Load and Cooling Load."
        ),
        "target_column": "Y1",
        "feature_descriptions": {
            "X1": "Relative Compactness",
            "X2": "Surface Area",
            "X3": "Wall Area",
            "X4": "Roof Area",
            "X5": "Overall Height",
            "X6": "Orientation",
            "X7": "Glazing Area",
            "X8": "Glazing Area Distribution",
        },
        "target_description": "Heating Load (kWh/m²)",
    },
    243: {
        "openml_data_id": 44957,
        "name": "Yacht Hydrodynamics",
        "repository_url": "https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics",
        "abstract": (
            "Prediction of residuary resistance of sailing yachts from hull geometry "
            "coefficients and Froude number. 308 instances, 6 features."
        ),
        "target_column": "RRPD",
        "feature_descriptions": {
            "LC": "Longitudinal position of the center of buoyancy",
            "PC": "Prismatic coefficient",
            "LDR": "Length-displacement ratio",
            "BDR": "Beam-draught ratio",
            "LBR": "Length-beam ratio",
            "Fr": "Froude number",
        },
        "target_description": "Residuary resistance per unit weight of displacement",
    },
    291: {
        "openml_data_id": 44958,
        "name": "Airfoil Self-Noise",
        "repository_url": "https://archive.ics.uci.edu/dataset/291/airfoil+self-noise",
        "abstract": (
            "NASA aerodynamic and acoustic tests of two- and three-dimensional airfoil "
            "blade sections. 1503 instances, 5 features."
        ),
        "target_column": "SSPL",
        "feature_descriptions": {
            "Frequency": "Frequency in Hertz",
            "AOA": "Angle of attack in degrees",
            "ChordLength": "Chord length in meters",
            "FSV": "Free-stream velocity in m/s",
            "SSDT": "Suction-side displacement thickness in meters",
        },
        "target_description": "Scaled sound pressure level (decibels)",
    },
    294: {
        "openml_data_id": 44955,
        "name": "Combined Cycle Power Plant",
        "repository_url": "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
        "abstract": (
            "Data collected from a Combined Cycle Power Plant over 6 years. "
            "9568 instances, 4 ambient condition features."
        ),
        "target_column": "PE",
        "feature_descriptions": {
            "AT": "Ambient Temperature (°C)",
            "V": "Exhaust Vacuum (cm Hg)",
            "AP": "Ambient Pressure (millibar)",
            "RH": "Relative Humidity (%)",
        },
        "target_description": "Net hourly electrical energy output (MW)",
    },
    477: {
        "openml_data_id": 44956,
        "name": "Real Estate Valuation",
        "repository_url": "https://archive.ics.uci.edu/dataset/477/real+estate+valuation",
        "abstract": (
            "Real estate valuation in Sindian District, New Taipei City, Taiwan. "
            "414 instances, 6 features."
        ),
        "target_column": "Y house price of unit area",
        "feature_descriptions": {
            "X1 transaction date": "Transaction date",
            "X2 house age": "House age (years)",
            "X3 distance to the nearest MRT station": "Distance to nearest MRT station (m)",
            "X4 number of convenience stores": "Number of convenience stores nearby",
            "X5 latitude": "Latitude (°)",
            "X6 longitude": "Longitude (°)",
        },
        "target_description": "House price of unit area (10k NTD / ping)",
    },
    504: {
        "openml_data_id": 44970,
        "name": "QSAR Fish Toxicity",
        "repository_url": "https://archive.ics.uci.edu/dataset/504/qsar+fish+toxicity",
        "abstract": (
            "Values for 6 molecular descriptors of 908 chemicals used to predict "
            "quantitative acute aquatic toxicity towards the fish Pimephales promelas."
        ),
        "target_column": "LC50",
        "feature_descriptions": {
            "CIC0": "information indices",
            "SM1_Dz": "2D matrix-based descriptors",
            "GATS1i": "2D autocorrelations",
            "NdsCH": "atom-type counts",
            "NdssC": "atom-type counts",
            "MLOGP": "molecular properties",
        },
        "target_description": "quantitative response, LC50 [-LOG(mol/L)]",
    },
    551: {
        "openml_data_id": 42974,
        "name": "Gas Turbine CO and NOx Emission Data Set",
        "repository_url": "https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set",
        "abstract": (
            "Hourly sensor readings from a gas turbine used to predict CO and NOx "
            "emissions. This fallback covers the 2011 shard (~7 400 rows) of the "
            "full 5-year UCI dataset (~36 700 rows)."
        ),
        "target_column": "CO",
        "feature_descriptions": {
            "AT": "Ambient temperature (°C)",
            "AP": "Ambient pressure (mbar)",
            "AH": "Ambient humidity (%)",
            "AFDP": "Air filter difference pressure (mbar)",
            "GTEP": "Gas turbine exhaust pressure (mbar)",
            "TIT": "Turbine inlet temperature (°C)",
            "TAT": "Turbine after temperature (°C)",
            "CDP": "Compressor discharge pressure (mbar)",
            "TEY": "Turbine energy yield (MWH)",
        },
        "target_description": "Carbon monoxide (CO) concentration (mg/m³)",
    },
    44978: {
        "openml_data_id": 44978,
        "name": "CPU Activity",
        "repository_url": "https://www.openml.org/d/44978",
        "abstract": (
            "DELVE benchmark dataset measuring 21 system-activity counters on 8 192 Unix "
            "workload snapshots. Task: predict the fraction of CPU time spent in user mode."
        ),
        "target_column": "usr",
        "feature_descriptions": {
            "lread": "Reads (transfers per second) by the disk subsystem",
            "lwrite": "Writes (transfers per second) by the disk subsystem",
            "scall": "Number of system calls of all types per second",
            "sread": "Number of read system calls per second",
            "swrite": "Number of write system calls per second",
            "fork": "Number of fork system calls per second",
            "exec": "Number of exec system calls per second",
            "rchar": "Number of characters transferred by read system calls per second",
            "wchar": "Number of characters transferred by write system calls per second",
            "pgout": "Number of page out requests per second",
            "ppgout": "Number of pages paged out per second",
            "pgfree": "Number of pages placed on the free list per second",
            "pgscan": "Number of pages checked if they can be freed per second",
            "atch": "Number of page attaches (returning a reclaimed page) per second",
            "pgin": "Number of page in requests per second",
            "ppgin": "Number of pages paged in per second",
            "pflt": "Number of page faults caused by protection errors per second",
            "vflt": "Number of page faults caused by address translation per second",
            "runqsz": "Process run queue size (processes waiting for CPU)",
            "freemem": "Number of memory pages available to user processes",
            "freeswap": "Number of disk blocks available for page swapping",
        },
        "target_description": "Fraction of CPU time spent in user mode (0–100)",
    },
    44980: {
        "openml_data_id": 44980,
        "name": "kin8nm",
        "repository_url": "https://www.openml.org/d/44980",
        "abstract": (
            "Forward kinematics benchmark for an 8-link robot arm. The 8nm variant is "
            "highly nonlinear and medium noisy, with 8,192 rows and 8 continuous inputs."
        ),
        "target_column": "y",
        "feature_descriptions": {
            "x1": "Joint-angle input 1 of the 8-link robot arm",
            "x2": "Joint-angle input 2 of the 8-link robot arm",
            "x3": "Joint-angle input 3 of the 8-link robot arm",
            "x4": "Joint-angle input 4 of the 8-link robot arm",
            "x5": "Joint-angle input 5 of the 8-link robot arm",
            "x6": "Joint-angle input 6 of the 8-link robot arm",
            "x7": "Joint-angle input 7 of the 8-link robot arm",
            "x8": "Joint-angle input 8 of the 8-link robot arm",
        },
        "target_description": "Forward-kinematics output for the 8-link robot arm",
    },
    560: {
        "openml_data_id": 46297,
        "name": "Seoul Bike Sharing Demand",
        "repository_url": "https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand",
        "abstract": (
            "Hourly count of rental bikes in Seoul's public bike-sharing system along "
            "with weather and holiday information. 8 760 instances."
        ),
        "target_column": "Rented Bike Count",
        "feature_descriptions": {
            "Date": "Date (yyyy-mm-dd)",
            "Hour": "Hour of the day (0–23)",
            "Temperature(C)": "Temperature (°C)",
            "Humidity(%)": "Relative humidity (%)",
            "Wind speed (m/s)": "Wind speed (m/s)",
            "Visibility (10m)": "Visibility (10 m)",
            "Dew point temperature(C)": "Dew point temperature (°C)",
            "Solar Radiation (MJ/m2)": "Solar radiation (MJ/m²)",
            "Rainfall(mm)": "Rainfall (mm)",
            "Snowfall (cm)": "Snowfall (cm)",
            "Seasons": "Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)",
            "Holiday": "Holiday flag (Holiday / No Holiday)",
            "Functioning Day": "Functioning day flag (Yes/No)",
        },
        "target_description": "Number of bikes rented per hour",
    },
}


class UCILoader:
    """Loads and pre-processes tabular datasets from UCI ML Repository and OpenML.

    Supports three lookup strategies tried in order:

    1. **Special datasets** — a small set of negative IDs mapped to built-in
       loaders (e.g. California Housing via scikit-learn).
    2. **Direct OpenML** — a curated list of positive IDs that are fetched
       directly from OpenML without going through ucimlrepo.
    3. **UCI via ucimlrepo** — fetched from the UCI repository, with an
       automatic OpenML fallback for datasets that are temporarily unavailable.
    """

    @staticmethod
    def _log_progress(message: str) -> None:
        logger.info("[UCI loader] %s", message)

    @staticmethod
    def _numeric_series(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    @classmethod
    def _is_numeric_series(cls, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series):
            return True
        coerced = cls._numeric_series(series)
        return coerced.notna().sum() >= max(3, int(0.5 * len(series)))

    @staticmethod
    def _format_number(value: float) -> str:
        if np.isfinite(value) and abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"

    @classmethod
    def _range_str(cls, series: pd.Series) -> str:
        if not cls._is_numeric_series(series):
            return "-"
        values = cls._numeric_series(series).dropna()
        if values.empty:
            return "-"
        return (
            f"[{cls._format_number(float(values.min()))} - "
            f"{cls._format_number(float(values.max()))}]"
        )

    @classmethod
    def _stats_str(cls, series: pd.Series) -> str:
        unique_count = int(series.dropna().nunique())
        return f"{cls._range_str(series)} | unique={unique_count}"

    def _print_dataframe_overview(
        self,
        *,
        metadata: Dict[str, Any],
        df: Optional[pd.DataFrame],
        feature_cols: List[str],
        target_cols: List[str],
        declared_type: Dict[str, str],
        width: int = 92,
        max_features: int | None = None,
    ) -> None:
        """Print a formatted dataset overview for a concrete feature/target split."""
        if df is not None:
            df = df.copy().replace(r"^\s*$", pd.NA, regex=True)
            rows_with_nans = int(df.isna().any(axis=1).sum())
            n_rows = int(len(df))
        else:
            rows_with_nans = None
            n_rows = int(metadata.get("num_instances") or 0)

        name = metadata.get("name", "Unknown dataset")
        abstract = (metadata.get("abstract") or "").strip()
        desc = shorten(" ".join(abstract.split()), width=width - 4, placeholder="…") if abstract else ""
        repo = metadata.get("repository_url")
        line = "─" * width

        print(line)
        print(name)
        if desc:
            print(f"  {desc}")
        if repo:
            print(f"  Repo: {repo}")
        print(line)

        if n_rows:
            if rows_with_nans is None:
                mv = metadata.get("has_missing_values")
                print(f"Rows: {n_rows:,}   |   Rows with NaNs: (data not available; metadata says: {mv})")
            else:
                print(f"Rows: {n_rows:,}   |   Rows with NaNs: {rows_with_nans:,}")
        else:
            print("Rows: (unknown)")

        print(f"Features: {len(feature_cols)}   |   Targets: {', '.join(target_cols) if target_cols else '(unknown)'}")
        print(line)

        def build_rows(columns: List[str], *, add_indices: bool = False) -> List[Tuple[str, str, str, str]]:
            rows: List[Tuple[str, str, str, str]] = []
            for index, column_name in enumerate(columns):
                display_name = f"[{index}] {column_name}" if add_indices else column_name
                if df is None or column_name not in df.columns:
                    type_name = declared_type.get(column_name, "unknown")
                    rows.append((display_name, type_name, "unknown", "-"))
                    continue
                series = df[column_name]
                type_name = declared_type.get(column_name, str(series.dtype))
                flag = "NUM" if self._is_numeric_series(series) else "NON-NUM"
                rows.append((display_name, type_name, flag, self._stats_str(series)))
            return rows

        def print_table(title: str, rows: List[Tuple[str, str, str, str]], limit: int | None = None) -> None:
            if not rows:
                return
            shown = rows if limit is None else rows[:limit]
            hidden = 0 if limit is None else max(0, len(rows) - len(shown))

            name_w = min(max(len(a) for a, _, _, _ in shown), 40)
            type_w = min(max(len(b) for _, b, _, _ in shown), 26)
            flag_w = 7

            print(title)
            print(
                f"{'Name'.ljust(name_w)}  {'Type'.ljust(type_w)}  "
                f"{'Flag'.ljust(flag_w)}  Range / Unique"
            )
            print(f"{'-'*name_w}  {'-'*type_w}  {'-'*flag_w}  {'-'*28}")
            for a, b, f, r in shown:
                print(f"{a.ljust(name_w)}  {b.ljust(type_w)}  {f.ljust(flag_w)}  {r}")
            if hidden:
                print(f"... ({hidden} more)")
            print(line)

        print_table("Features", build_rows(feature_cols, add_indices=True), limit=max_features)
        print_table("Targets", build_rows(target_cols, add_indices=False), limit=None)

    def load_uci_dataset(
        self,
        id: int,
        *,
        show_summary: bool = False,
    ) -> Dict[str, Any]:
        """Load a regression dataset by ID, returning a standardised dict.

        Tries the following sources in order: special built-in loaders (negative
        IDs), direct OpenML datasets, UCI via ucimlrepo with an OpenML fallback.

        Args:
            id: Dataset identifier. Negative IDs map to built-in loaders;
                certain positive IDs are fetched directly from OpenML; all
                remaining positive IDs are fetched from the UCI repository.
            show_summary: If ``True``, print a formatted overview table after
                loading.

        Returns:
            A dict with keys:
                - ``"data"``: sub-dict with ``"features"`` (DataFrame),
                  ``"targets"`` (DataFrame), and ``"original"`` (DataFrame).
                - ``"metadata"``: dataset metadata dict.
                - ``"variables"``: DataFrame describing each column.

        Raises:
            OSError: If the dataset cannot be fetched from its configured source
                and no fallback is available.
            DatasetNotFoundError: If the UCI ID is not recognised and no
                OpenML fallback is configured.
        """
        special_dataset = self._load_special_dataset(id=id)
        if special_dataset is not None:
            if show_summary:
                self.print_uci_summary(special_dataset)
            return special_dataset

        direct_openml_dataset = self._load_direct_openml_dataset(id=id)
        if direct_openml_dataset is not None:
            if show_summary:
                self.print_uci_summary(direct_openml_dataset)
            return direct_openml_dataset

        fetch_start = time.perf_counter()
        self._log_progress(f"Fetching dataset id={id} via ucimlrepo...")
        try:
            dataset = fetch_ucirepo(id=id)
        except (OSError, DatasetNotFoundError) as exc:
            self._log_progress(
                f"ucimlrepo fetch failed after {time.perf_counter() - fetch_start:.2f}s: {exc}"
            )
            dataset = self._load_openml_fallback_dataset(id=id, source_error=exc)
            if dataset is None:
                self._log_progress(f"No configured fallback for dataset id={id}; re-raising error.")
                raise
            self._log_progress(f"OpenML fallback loaded dataset id={id}.")
        else:
            self._log_progress(
                f"ucimlrepo fetch complete in {time.perf_counter() - fetch_start:.2f}s."
            )

        if show_summary:
            self.print_uci_summary(dataset)
        return dataset

    def _load_direct_openml_dataset(
        self,
        *,
        id: int,
        source_error: Exception | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a dataset configured as a direct OpenML data ID.

        Args:
            id: Dataset ID to look up in ``_DIRECT_OPENML_DATASET_SPECS``.
            source_error: Upstream error that triggered this call, if any.
                Included in the raised ``RuntimeError`` message.

        Returns:
            A standardised dataset dict, or ``None`` if *id* is not in the
            direct-OpenML spec table.

        Raises:
            RuntimeError: If the OpenML fetch itself fails.
        """
        spec = _DIRECT_OPENML_DATASET_SPECS.get(id)
        if spec is None:
            return None

        openml_fetch_start = time.perf_counter()
        self._log_progress(f"Fetching direct OpenML dataset data_id={id}...")
        try:
            bunch = fetch_openml(data_id=id, as_frame=True, parser="auto")
        except Exception as exc:
            context = (
                f"after ucimlrepo failed with: {source_error}"
                if source_error is not None
                else "via the direct OpenML loader"
            )
            raise RuntimeError(
                f"Dataset id={id} is configured as a direct OpenML dataset, but "
                f"OpenML loading failed {context}."
            ) from exc
        self._log_progress(
            f"Direct OpenML fetch complete in {time.perf_counter() - openml_fetch_start:.2f}s."
        )

        frame = getattr(bunch, "frame", None)
        features = getattr(bunch, "data", None)
        target = getattr(bunch, "target", None)

        targets_df: Optional[pd.DataFrame] = None
        if isinstance(target, pd.Series):
            target_name = target.name or str(spec.get("target_column", "target"))
            targets_df = target.rename(target_name).to_frame()
        elif isinstance(target, pd.DataFrame):
            if not target.empty:
                targets_df = target.copy()
        elif target is not None:
            target_name = str(spec.get("target_column", "target"))
            targets_df = pd.DataFrame({target_name: np.asarray(target)})

        if not isinstance(frame, pd.DataFrame):
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(
                    features,
                    columns=list(getattr(bunch, "feature_names", []) or []),
                )
            frame = features.copy()
            if targets_df is not None:
                frame = pd.concat([frame, targets_df], axis=1)

        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"OpenML dataset data_id={id} did not yield a DataFrame.")

        if targets_df is None or targets_df.shape[1] == 0:
            hinted_target = spec.get("target_column")
            if hinted_target is not None and str(hinted_target) in frame.columns:
                targets_df = frame[[str(hinted_target)]].copy()
            elif frame.shape[1] >= 2:
                fallback_target_name = str(frame.columns[-1])
                self._log_progress(
                    f"Direct OpenML dataset data_id={id} did not expose a target column; "
                    f"using trailing column '{fallback_target_name}' as the initial target."
                )
                targets_df = frame[[fallback_target_name]].copy()
            else:
                raise ValueError(
                    f"OpenML dataset data_id={id} does not expose a usable target column."
                )

        target_names = [str(column_name) for column_name in targets_df.columns]
        original_df = frame.copy()
        missing_targets = [name for name in target_names if name not in original_df.columns]
        if missing_targets:
            original_df = pd.concat([original_df, targets_df], axis=1)

        features_df = original_df.drop(columns=target_names, errors="ignore").copy()
        targets_df = targets_df.copy()
        feature_descriptions = dict(spec.get("feature_descriptions", {}))

        variable_rows: List[Dict[str, Any]] = []
        for column_name in features_df.columns:
            variable_rows.append({
                "name": str(column_name),
                "role": "Feature",
                "type": self._infer_variable_type(features_df[column_name]),
                "description": feature_descriptions.get(str(column_name), ""),
                "missing_values": bool(features_df[column_name].isna().any()),
            })
        for column_name in targets_df.columns:
            variable_rows.append({
                "name": str(column_name),
                "role": "Target",
                "type": self._infer_variable_type(targets_df[column_name]),
                "description": str(spec.get("target_description", "")),
                "missing_values": bool(targets_df[column_name].isna().any()),
            })

        details = dict(getattr(bunch, "details", {}) or {})
        metadata = {
            "dataset_id": id,
            "uci_id": None,
            "name": str(spec.get("name") or details.get("name") or getattr(bunch, "name", id)),
            "abstract": str(spec.get("abstract") or details.get("description") or ""),
            "repository_url": str(spec.get("repository_url") or f"https://www.openml.org/d/{id}"),
            "data_url": details.get("url") or details.get("original_data_url"),
            "num_instances": int(original_df.shape[0]),
            "target_col": target_names,
            "has_missing_values": bool(original_df.isna().any(axis=1).any()),
            "fallback_source": "openml_direct",
            "fallback_reason": str(source_error) if source_error is not None else None,
            "openml_data_id": id,
            "openml_dataset_name": details.get("name", getattr(bunch, "name", None)),
            "openml_dataset_version": details.get("version"),
        }

        self._log_progress(
            f"Direct OpenML dataset data_id={id} ready with {original_df.shape[0]:,} rows, "
            f"{features_df.shape[1]} feature columns, targets={target_names}."
        )
        return {
            "data": {
                "ids": None,
                "features": features_df,
                "targets": targets_df,
                "original": original_df,
                "headers": original_df.columns,
            },
            "metadata": metadata,
            "variables": pd.DataFrame.from_records(variable_rows),
        }

    def _load_special_dataset(self, *, id: int) -> Optional[Dict[str, Any]]:
        """Load a dataset from a built-in non-UCI source (e.g. scikit-learn).

        Args:
            id: Dataset ID to look up in ``_SPECIAL_DATASET_SPECS``.

        Returns:
            A standardised dataset dict, or ``None`` if *id* is not a
            special dataset.

        Raises:
            ValueError: If *id* maps to a special spec but no loader is
                implemented for it.
        """
        spec = _SPECIAL_DATASET_SPECS.get(id)
        if spec is None:
            return None

        fetch_start = time.perf_counter()
        loader_name = str(spec.get("loader_name", "special_loader"))
        self._log_progress(f"Fetching dataset id={id} via {loader_name}...")
        if id == CALIFORNIA_HOUSING_DATASET_ID:
            bunch = fetch_california_housing(as_frame=True)
        else:
            raise ValueError(f"Unsupported special dataset id={id}.")
        self._log_progress(
            f"{loader_name} fetch complete in {time.perf_counter() - fetch_start:.2f}s."
        )

        frame = getattr(bunch, "frame", None)
        if not isinstance(frame, pd.DataFrame):
            features = getattr(bunch, "data", None)
            target = getattr(bunch, "target", None)
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(
                    features,
                    columns=list(getattr(bunch, "feature_names", []) or []),
                )
            if isinstance(target, pd.Series):
                target_name = target.name or str(spec["target_column"])
                target_df = target.rename(target_name).to_frame()
            else:
                target_name = str(spec["target_column"])
                target_df = pd.DataFrame({target_name: np.asarray(target)})
            frame = pd.concat([features, target_df], axis=1)

        target_name = str(spec["target_column"])
        original_df = frame.copy()
        features_df = original_df.drop(columns=[target_name]).copy()
        targets_df = original_df[[target_name]].copy()
        feature_descriptions = dict(spec.get("feature_descriptions", {}))

        variable_rows: List[Dict[str, Any]] = []
        for column_name in features_df.columns:
            variable_rows.append({
                "name": str(column_name),
                "role": "Feature",
                "type": self._infer_variable_type(features_df[column_name]),
                "description": feature_descriptions.get(str(column_name), ""),
                "missing_values": bool(features_df[column_name].isna().any()),
            })
        variable_rows.append({
            "name": target_name,
            "role": "Target",
            "type": self._infer_variable_type(targets_df[target_name]),
            "description": str(spec.get("target_description", "")),
            "missing_values": bool(targets_df[target_name].isna().any()),
        })

        metadata = {
            "dataset_id": id,
            "uci_id": None,
            "name": str(spec["name"]),
            "abstract": str(spec.get("abstract", "")),
            "repository_url": str(spec.get("repository_url", "")),
            "data_url": None,
            "num_instances": int(original_df.shape[0]),
            "target_col": [target_name],
            "has_missing_values": bool(original_df.isna().any(axis=1).any()),
            "fallback_source": "sklearn",
            "fallback_reason": None,
            "source_loader": loader_name,
        }
        self._log_progress(
            f"Special dataset id={id} ready with {original_df.shape[0]:,} rows, "
            f"{features_df.shape[1]} feature columns, target='{target_name}'."
        )
        return {
            "data": {
                "ids": None,
                "features": features_df,
                "targets": targets_df,
                "original": original_df,
                "headers": original_df.columns,
            },
            "metadata": metadata,
            "variables": pd.DataFrame.from_records(variable_rows),
        }

    @staticmethod
    def _infer_variable_type(series: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(series):
            return "Integer"
        if pd.api.types.is_numeric_dtype(series):
            return "Continuous"
        return "Categorical"

    def _load_openml_fallback_dataset(
        self,
        *,
        id: int,
        source_error: Exception,
    ) -> Optional[Dict[str, Any]]:
        """Fetch an OpenML fallback for a UCI dataset that failed to load.

        Args:
            id: UCI dataset ID that could not be fetched.
            source_error: The exception raised by the upstream UCI fetch.

        Returns:
            A standardised dataset dict, or ``None`` if no OpenML fallback is
            configured for *id*.

        Raises:
            RuntimeError: If both the UCI fetch and the OpenML fallback fail.
        """
        fallback_spec = _OPENML_UCI_FALLBACKS.get(id)
        if fallback_spec is None:
            return None

        openml_data_id = int(fallback_spec["openml_data_id"])
        openml_fetch_start = time.perf_counter()
        self._log_progress(
            f"Fetching OpenML fallback data_id={openml_data_id} for UCI dataset id={id}..."
        )
        try:
            bunch = fetch_openml(data_id=openml_data_id, as_frame=True, parser="auto")
        except Exception as exc:
            raise RuntimeError(
                f"UCI dataset id={id} could not be loaded via ucimlrepo, and the "
                f"OpenML fallback data_id={openml_data_id} also failed."
            ) from exc
        self._log_progress(
            f"OpenML fetch complete in {time.perf_counter() - openml_fetch_start:.2f}s."
        )

        frame = getattr(bunch, "frame", None)
        if not isinstance(frame, pd.DataFrame):
            features = getattr(bunch, "data", None)
            target = getattr(bunch, "target", None)
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(
                    features,
                    columns=list(getattr(bunch, "feature_names", []) or []),
                )
            if isinstance(target, pd.Series):
                target_name = target.name or str(fallback_spec["target_column"])
                target_df = target.rename(target_name).to_frame()
            else:
                target_name = str(fallback_spec["target_column"])
                target_df = pd.DataFrame({target_name: np.asarray(target)})
            frame = pd.concat([features, target_df], axis=1)

        target_name = str(fallback_spec["target_column"])
        inferred_target_name = getattr(getattr(bunch, "target", None), "name", None)
        if target_name not in frame.columns and inferred_target_name in frame.columns:
            target_name = str(inferred_target_name)
        if target_name not in frame.columns:
            raise ValueError(
                f"OpenML fallback for UCI dataset id={id} is missing target column "
                f"'{fallback_spec['target_column']}'."
            )

        original_df = frame.copy()
        features_df = original_df.drop(columns=[target_name]).copy()
        targets_df = original_df[[target_name]].copy()
        feature_descriptions = dict(fallback_spec.get("feature_descriptions", {}))

        variable_rows: List[Dict[str, Any]] = []
        for column_name in features_df.columns:
            variable_rows.append({
                "name": str(column_name),
                "role": "Feature",
                "type": self._infer_variable_type(features_df[column_name]),
                "description": feature_descriptions.get(str(column_name), ""),
                "missing_values": bool(features_df[column_name].isna().any()),
            })
        variable_rows.append({
            "name": target_name,
            "role": "Target",
            "type": self._infer_variable_type(targets_df[target_name]),
            "description": str(fallback_spec.get("target_description", "")),
            "missing_values": bool(targets_df[target_name].isna().any()),
        })

        details = dict(getattr(bunch, "details", {}) or {})
        metadata = {
            "uci_id": id,
            "name": str(fallback_spec["name"]),
            "abstract": str(fallback_spec.get("abstract", "")),
            "repository_url": str(fallback_spec.get("repository_url", "")),
            "data_url": details.get("url") or details.get("original_data_url"),
            "num_instances": int(original_df.shape[0]),
            "target_col": [target_name],
            "has_missing_values": bool(original_df.isna().any(axis=1).any()),
            "fallback_source": "openml",
            "fallback_reason": str(source_error),
            "openml_data_id": openml_data_id,
            "openml_dataset_name": details.get("name", getattr(bunch, "name", None)),
            "openml_dataset_version": details.get("version"),
        }

        self._log_progress(
            f"UCI dataset id={id} unavailable via ucimlrepo; "
            f"using OpenML fallback data_id={openml_data_id}."
        )
        self._log_progress(
            f"Fallback frame ready with {original_df.shape[0]:,} rows, "
            f"{features_df.shape[1]} feature columns, target='{target_name}'."
        )
        return {
            "data": {
                "ids": None,
                "features": features_df,
                "targets": targets_df,
                "original": original_df,
                "headers": original_df.columns,
            },
            "metadata": metadata,
            "variables": pd.DataFrame.from_records(variable_rows),
        }

    def print_uci_summary(
        self,
        uci: Dict[str, Any],
        *,
        width: int = 92,
        max_features: int | None = None,
    ) -> None:
        """Print a formatted terminal summary for a UCI dataset dict.

        Args:
            uci: Dataset dict as returned by `load_uci_dataset`.
            width: Terminal width in characters.
            max_features: If set, truncate the feature table after this many
                rows and print a ``"... (N more)"`` footer.
        """
        md = (uci.get("metadata") or {})
        variables = uci.get("variables", None)

        data = uci.get("data", {}) or {}
        X = data.get("features", None)
        y = data.get("targets", None)
        original = data.get("original", None)
        headers = data.get("headers", None)

        if isinstance(y, pd.Series):
            y = y.to_frame()
        if y is not None and not isinstance(y, pd.DataFrame):
            try:
                y = pd.DataFrame(y)
            except (ValueError, TypeError):
                y = None

        df: Optional[pd.DataFrame] = None
        if isinstance(original, pd.DataFrame):
            df = original.copy()
        elif isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            df = pd.concat([X, y], axis=1)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()

        if headers is None and df is not None:
            headers = df.columns
        headers_list = list(headers) if headers is not None else (list(df.columns) if df is not None else [])

        target_cols: List[str] = []
        feature_cols: List[str] = []
        if isinstance(variables, pd.DataFrame) and {"name", "role"}.issubset(variables.columns):
            for _, r in variables.iterrows():
                col = str(r["name"])
                role = str(r["role"]).strip().lower()
                if role == "target":
                    target_cols.append(col)
                elif role == "feature":
                    feature_cols.append(col)

            if headers_list:
                order = {c: i for i, c in enumerate(headers_list)}
                target_cols.sort(key=lambda c: order.get(c, 10**9))
                feature_cols.sort(key=lambda c: order.get(c, 10**9))
        else:
            tc = md.get("target_col") or []
            target_cols = [tc] if isinstance(tc, str) else list(tc)
            feature_cols = [c for c in headers_list if c not in set(target_cols)]

        declared_type: Dict[str, str] = {}
        if isinstance(variables, pd.DataFrame) and {"name", "type"}.issubset(variables.columns):
            declared_type = {str(r["name"]): str(r["type"]) for _, r in variables.iterrows()}

        self._print_dataframe_overview(
            metadata=md,
            df=df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            declared_type=declared_type,
            width=width,
            max_features=max_features,
        )

    def refine_dataset(
        self,
        uci: Any,
        *,
        target_column: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
        """Interactively select features and a target, remove NaN rows, and return arrays.

        Prompts the user to choose feature columns by index and a target column,
        then drops any row containing NaN or empty values and returns
        JAX-compatible float32 arrays.

        Args:
            uci: Dataset dict as returned by `load_uci_dataset`.
            target_column: Optional suggested target column name. If provided and
                found in the dataset, it becomes the default selection.

        Returns:
            A tuple ``(X, y, feature_names, target_name)`` where *X* has shape
            ``(n, d)`` and *y* has shape ``(n,)``, both as float32 numpy arrays.

        Raises:
            ValueError: If the feature or target tables are empty, or if no rows
                remain after NaN removal.
        """
        refine_start = time.perf_counter()

        def _get(container: Any, key: str, default: Any = None) -> Any:
            if container is None:
                return default
            if isinstance(container, dict):
                return container.get(key, default)
            return getattr(container, key, default)

        def _parse_feature_ids(raw: str, max_index: int) -> List[int]:
            raw = raw.strip().lower()
            if raw in {"", "all", "*"}:
                return list(range(max_index))

            selected: List[int] = []
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                if "-" in token:
                    parts = token.split("-", 1)
                    if len(parts) != 2:
                        raise ValueError(f"Invalid range token: '{token}'")
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    if start > end:
                        raise ValueError(f"Invalid range '{token}' (start > end)")
                    selected.extend(list(range(start, end + 1)))
                else:
                    selected.append(int(token))

            deduped: List[int] = []
            seen: set = set()
            for idx in selected:
                if idx not in seen:
                    seen.add(idx)
                    deduped.append(idx)

            if not deduped:
                raise ValueError("No feature IDs selected.")
            if min(deduped) < 0 or max(deduped) >= max_index:
                raise ValueError(f"Feature IDs must be in [0, {max_index - 1}].")
            return deduped

        def _resolve_target_selector(
            raw: str,
            *,
            feature_names: List[str],
            target_names: List[str],
        ) -> tuple[str, str]:
            token = raw.strip()
            if not token:
                raise ValueError("Target selection cannot be empty here.")

            lowered = token.lower()
            prefix_to_names = {
                "f": feature_names,
                "feature": feature_names,
                "x": feature_names,
                "t": target_names,
                "target": target_names,
                "y": target_names,
            }
            for prefix, names in prefix_to_names.items():
                marker = f"{prefix}:"
                if lowered.startswith(marker):
                    index_str = token[len(marker):].strip()
                    try:
                        index = int(index_str)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid target selector '{token}'. Expected e.g. '{prefix}:0'."
                        ) from exc
                    if index < 0 or index >= len(names):
                        raise ValueError(
                            f"Target selector '{token}' is out of range for prefix '{prefix}'."
                        )
                    return ("feature" if names is feature_names else "target"), names[index]

            if token in feature_names:
                return "feature", token
            if token in target_names:
                return "target", token

            try:
                feature_index = int(token)
            except ValueError as exc:
                raise ValueError(
                    "Unknown target selection. Use a feature ID like '1', "
                    "a prefixed selector like 't:0', or an exact column name."
                ) from exc

            if feature_index < 0 or feature_index >= len(feature_names):
                raise ValueError(
                    f"Feature-style target index must be in [0, {len(feature_names) - 1}]."
                )
            return "feature", feature_names[feature_index]

        def _resolve_default_target(
            *,
            suggested_name: str | None,
            feature_names: List[str],
            target_names: List[str],
        ) -> tuple[str, str]:
            if suggested_name is not None:
                if suggested_name in feature_names:
                    return "feature", suggested_name
                if suggested_name in target_names:
                    return "target", suggested_name
                print(
                    f"Warning: suggested target column '{suggested_name}' not found "
                    f"(available columns: {feature_names + target_names}). "
                    "Falling back to default target selection."
                )
            if target_names:
                return "target", target_names[0]
            raise ValueError("No target candidates are available.")

        data = _get(uci, "data", {})
        X_raw = _get(data, "features", None)
        y_raw = _get(data, "targets", None)

        if X_raw is None:
            raise ValueError("Dataset has no 'features' data.")
        if y_raw is None:
            raise ValueError("Dataset has no 'targets' data.")

        X_df = X_raw.copy() if isinstance(X_raw, pd.DataFrame) else pd.DataFrame(X_raw)
        if isinstance(y_raw, pd.Series):
            y_df = y_raw.to_frame()
        elif isinstance(y_raw, pd.DataFrame):
            y_df = y_raw.copy()
        else:
            y_df = pd.DataFrame(y_raw)

        if X_df.shape[1] == 0:
            raise ValueError("Feature table is empty.")
        if y_df.shape[1] == 0:
            raise ValueError("Target table is empty.")

        feature_names = [str(c) for c in X_df.columns]
        target_names = [str(c) for c in y_df.columns]
        self._log_progress(
            f"Preparing refinement view with {len(feature_names)} feature candidates and "
            f"{len(target_names)} target candidates."
        )
        suggested_name = str(target_column) if target_column is not None else None
        default_source, default_target_name = _resolve_default_target(
            suggested_name=suggested_name,
            feature_names=feature_names,
            target_names=target_names,
        )

        print("\nAvailable target columns:")
        for index, name in enumerate(feature_names):
            print(f"  [f:{index}] {name}")
        for index, name in enumerate(target_names):
            print(f"  [t:{index}] {name}")

        default_selector_prefix = "f" if default_source == "feature" else "t"
        default_names = feature_names if default_source == "feature" else target_names
        default_index = default_names.index(default_target_name)
        prompt = (
            "\nSelect target column "
            f"(blank for {default_selector_prefix}:{default_index} / {default_target_name}; "
            "accepts feature ID like '1', prefixed selector like 't:0', or exact name): "
        )
        while True:
            raw_target_input = input(prompt)
            if not raw_target_input.strip():
                target_source, target_name = default_source, default_target_name
                break
            try:
                target_source, target_name = _resolve_target_selector(
                    raw_target_input,
                    feature_names=feature_names,
                    target_names=target_names,
                )
                break
            except ValueError as exc:
                print(f"Invalid input: {exc}")

        if target_source == "feature":
            selected_target = X_df[target_name].copy()
            X_df = X_df.drop(columns=[target_name])
            print(f"Selected target column from feature table: '{target_name}'")
        else:
            selected_target = y_df[target_name].copy()
            print(f"Selected target column from target table: '{target_name}'")
        self._log_progress(f"Target selection resolved to '{target_name}' from {target_source} table.")

        feature_names = [str(c) for c in X_df.columns]
        variables = _get(uci, "variables", None)
        metadata = (_get(uci, "metadata", {}) or {})
        declared_type: Dict[str, str] = {}
        if isinstance(variables, pd.DataFrame) and {"name", "type"}.issubset(variables.columns):
            declared_type = {str(r["name"]): str(r["type"]) for _, r in variables.iterrows()}

        selected_overview_df = pd.concat(
            [X_df, selected_target.rename(target_name)],
            axis=1,
        )
        print("\nSelected target overview")
        self._print_dataframe_overview(
            metadata=metadata,
            df=selected_overview_df,
            feature_cols=feature_names,
            target_cols=[target_name],
            declared_type=declared_type,
        )

        while True:
            user_input = input("\nSelect feature IDs to include (e.g. 0,2,4-7 or 'all'): ")
            try:
                selected_ids = _parse_feature_ids(user_input, len(feature_names))
                break
            except ValueError as exc:
                print(f"Invalid input: {exc}")

        selected_feature_names = [feature_names[i] for i in selected_ids]
        self._log_progress(
            f"Selected {len(selected_feature_names)} feature(s); dropping NaN/empty rows next."
        )
        merged = pd.concat([X_df[selected_feature_names], selected_target], axis=1)
        merged = merged.replace(r"^\s*$", pd.NA, regex=True)

        rows_before = len(merged)
        merged = merged.dropna(axis=0, how="any")
        rows_after = len(merged)
        print(f"Removed {rows_before - rows_after} rows containing NaN/empty values.")

        if merged.empty:
            raise ValueError("No rows left after removing NaN/empty values.")

        X_clean = merged[selected_feature_names]
        y_clean = merged[target_name]

        self._log_progress(
            f"Encoding selected columns into numeric arrays for matrix shape "
            f"({len(X_clean):,}, {len(selected_feature_names)})."
        )
        encoded_cols: List[np.ndarray] = []
        encoded_feature_count = 0
        for col in selected_feature_names:
            series = X_clean[col]
            as_numeric = pd.to_numeric(series, errors="coerce")
            if as_numeric.notna().all():
                encoded_cols.append(as_numeric.to_numpy(dtype=np.float32))
            else:
                codes, _ = pd.factorize(series.astype(str))
                encoded_cols.append(codes.astype(np.float32))
                encoded_feature_count += 1

        X_np = np.column_stack(encoded_cols).astype(np.float32)

        y_numeric = pd.to_numeric(y_clean, errors="coerce")
        if y_numeric.notna().all():
            y_np = y_numeric.to_numpy(dtype=np.float32)
        else:
            y_codes, _ = pd.factorize(y_clean.astype(str))
            y_np = y_codes.astype(np.int32)

        if encoded_feature_count:
            print(f"Encoded {encoded_feature_count} non-numeric selected feature(s) as integer IDs.")
        if not y_numeric.notna().all():
            print("Encoded non-numeric target values as integer IDs.")

        self._log_progress(
            f"Refinement complete in {time.perf_counter() - refine_start:.2f}s. "
            f"Final shapes: X={X_np.shape}, y={y_np.shape}."
        )
        return X_np, y_np, selected_feature_names, target_name


Utils = UCILoader  # deprecated alias — use UCILoader

__all__ = ["UCILoader", "CALIFORNIA_HOUSING_DATASET_ID"]
