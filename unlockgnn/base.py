"""Wrappers for main functionality of model training and saving."""
from __future__ import annotations

import json
import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pymatgen
import sklearn
import tensorflow as tf
import tensorflow_probability as tfp
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel
from pyarrow import feather
from sklearn.metrics import mean_absolute_error
from tensorflow.python.framework.errors_impl import NotFoundError

from .datalib.preprocessing import LayerScaler
from .gp.gp_trainer import GPTrainer, convert_index_points
from .gp.kernel_layers import KernelLayer
from .gp.vgp_trainer import SingleLayerVGP
from .utilities.serialization import deserialize_array, serialize_array


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class GNN(Protocol):
    """Class for duck typing of generic GNNs."""

    def __init__(self, *args, **kwargs):
        ...
        # TODO: Flesh this out with the proper methods. Needs a more generalised LayerExtractor first.

    def train(self, *args, **kwargs):
        ...

    def save_model(self, *args, **kwargs):
        ...


class ProbGNN(ABC):
    """An abstract class for developing GNNs with uncertainty quantification.

    Provides a bundled interface for creating, training, saving and loading
    a GNN and a Gaussian process, minimising data handling for the end user.

    Args:
        train_structs: The training structures.
        train_targets: The training targets.
        val_structs: The validation structures.
        val_targets: The validation targets.
        gp_type: The method to use for the Gaussian process.
            Must be either 'GP' or 'VGP'.
        save_dir: The directory to save files to during training.
            Files include GNN and GP checkpoints.
        ntarget: The number of target variables.
            This can only be greater than one if `gp_type` is 'VGP'.
        layer_index: The index of the layer to extract outputs from
            within :attr:`gnn`. Defaults to the concatenation
            layer.
        num_inducing_points: The number of inducing points for the `VGP`.
            Can only be set for `gp_type='VGP'`.
        kernel: The kernel to use. Defaults to a radial basis function.
        training_stage: The stage of training the model is at.
            Only applies when loading a model.
        sf: The pre-calculated scaling factor. Only applicable when loading
            a pre-trained model.
        **kwargs: Keyword arguments to pass to :meth:`make_gnn`.

    Attributes:
        gnn: The GNN model.
        gp: The GP model.
        train_structs: The training structures.
        train_targets: The training targets.
        val_structs: The validation structures.
        val_targets: The validation targets.
        gp_type: The method to use for the Gaussian process.
            One of 'GP' or 'VGP'.
        save_dir: The directory to save files to during training.
            Files include GNN and GP checkpoints.
        ntarget: The number of target variables.
        layer_index: The index of the layer to extract outputs from
            within :attr:`gnn`.
        num_inducing_points: The number of inducing points for the `VGP`.
            Shoud be `None` for `gp_type='GP'`.
        kernel: The kernel to use. `None` means a radial basis function.
        sf: The scaling factor. Defaults to `None` when uncalculated.
        gnn_ckpt_path: The path to the GNN checkpoints.
        gnn_save_path: The path to the saved GNN.
        gp_ckpt_path: The path to the GP checkpoints.
        gp_save_path: The path to the saved GP.
        kernel_save_path: The path to the saved kernel.
        data_save_path: The path to the saved serialized data needed for
            reloading the GP: see :meth:`_gen_serial_data`.
        train_database: The path to the training database.
        val_database: The path to the validation database.
        sf_path: The path to the saved :attr:`sf`.
        meta_path: The path to the saved metadata: see :meth:`_write_metadata`.

    """

    def __init__(
        self,
        train_structs: List[pymatgen.Structure],
        train_targets: List[Union[np.ndarray, float]],
        val_structs: List[pymatgen.Structure],
        val_targets: List[Union[np.ndarray, float]],
        gp_type: Literal["GP", "VGP"],
        save_dir: Union[str, Path],
        ntarget: int = 1,
        layer_index: int = -4,
        num_inducing_points: Optional[int] = None,
        kernel: Optional[
            Union[tfp.math.psd_kernels.PositiveSemidefiniteKernel, KernelLayer]
        ] = None,
        training_stage: int = 0,
        sf: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initialize class."""
        if gp_type not in ["GP", "VGP"]:
            raise ValueError(f"`gp_type` must be one of 'GP' or 'VGP', got {gp_type=}")
        if gp_type == "GP":
            if ntarget > 1:
                raise NotImplementedError(
                    f"Can only have `ntarget > 1` when `gp_type` is 'VGP' (got {ntarget=})"
                )
            if num_inducing_points is not None:
                raise ValueError(
                    "`num_inducing_points` can only be set when `gp_type` is `VGP`"
                )
        if gp_type == "VGP":
            if num_inducing_points is None:
                raise ValueError(
                    "`num_inducing_points` must be supplied for `gp_type=VGP`, "
                    f"got {num_inducing_points=}"
                )

        self.gp_type = gp_type
        self.train_structs = train_structs
        self.train_targets = train_targets
        self.val_structs = val_structs
        self.val_targets = val_targets

        self.ntarget = ntarget
        self.sf = sf
        self.layer_index = layer_index
        self.num_inducing_points = num_inducing_points
        self.kernel = kernel

        self._validate_kernel()

        self.assign_save_directories(Path(save_dir))

        self.gnn: GNN = (
            self.make_gnn(**kwargs) if training_stage == 0 else self.load_gnn()
        )

        # Initialize GP
        if training_stage < 2:
            self.gp: Optional[Union[GPTrainer, SingleLayerVGP]] = None
        else:
            index_points = np.stack(self.get_index_points(self.train_structs))
            index_points = convert_index_points(index_points)

            if gp_type == "VGP":
                # Should already have been caught, but for the type checker's sake
                assert num_inducing_points is not None
                self.gp = SingleLayerVGP(
                    index_points,
                    num_inducing_points,
                    ntarget,
                    prev_model=str(self.gp_save_path),
                    kernel=self.kernel,
                )
                try:
                    self.gp.model.load_weights(str(self.gp_ckpt_path))
                except Exception as e:
                    print(f"Couldn't load any VGP checkpoints: {e}")

            else:
                targets = convert_index_points(np.stack(self.train_targets))
                self.gp = GPTrainer(
                    index_points, targets, self.gp_ckpt_path, self.kernel
                )

            self.kernel = self.gp.kernel

    def assign_save_directories(self, save_dir: Path) -> None:
        """Assign the directories for saving components.

        Also instantiates the base directory and the data directory.

        Args:
            save_dir: The base path for the save directory.

        """
        self.save_dir = save_dir

        self.gnn_ckpt_path = self.save_dir / "gnn_ckpts"
        self.gnn_save_path = self.save_dir / "gnn_model"
        self.gp_ckpt_path = self.save_dir / "gp_ckpts"
        self.gp_save_path = self.save_dir / "gp_model"
        self.kernel_save_path = self.save_dir / "kernel.pkl"

        self.data_save_path = self.save_dir / "data"
        self.train_database = self.data_save_path / "train.fthr"
        self.val_database = self.data_save_path / "val.fthr"
        self.sf_path = self.data_save_path / "sf.npy"
        self.meta_path = self.data_save_path / "meta.txt"

        # * Make directories
        for direct in [self.save_dir, self.data_save_path]:
            os.makedirs(direct, exist_ok=True)

    def _validate_kernel(self):
        """Validate the assigned kernel.

        Passes if kernel is yet to be assigned.

        """
        if self.kernel is None:
            return

        expected_kernels = {
            "VGP": KernelLayer,
            "GP": tfp.math.psd_kernels.PositiveSemidefiniteKernel,
        }
        current_expected_kernel = expected_kernels[self.gp_type]

        if not isinstance(self.kernel, current_expected_kernel):
            raise TypeError(
                f"Expected kernel with type {current_expected_kernel} for {self.gp_type=}, got {type(self.kernel)}"
            )

    @abstractmethod
    def make_gnn(self, **kwargs) -> GNN:
        """Construct a new GNN."""
        raise NotImplementedError()

    @abstractmethod
    def load_gnn(self) -> GNN:
        """Load a pre-trained GNN."""
        raise NotImplementedError()

    @property
    def training_stage(self) -> Literal[0, 1, 2]:
        """Indicate the training stage the model is at.

        Returns:
            training_stage: How much of the model is trained.
                Can take one of three values:

                * 0 - Untrained.
                * 1 - :attr:`gnn` trained.
                * 2 - :attr:`gnn` and :attr:`gp` trained.

        """
        return int(self.gnn_save_path.exists()) + bool(self.gp)  # type: ignore

    @abstractmethod
    def train_gnn(self):
        """Train the GNN."""
        pass

    @abstractmethod
    def save_gnn(self):
        """Save the GNN."""
        pass

    def _update_sf(self):
        """Update the saved scaling factor.

        This must be called to update :attr:`sf` whenever the MEGNetModel
        is updated (i.e. trained).

        """
        ls = LayerScaler.from_train_data(
            self.gnn, self.train_structs, layer_index=self.layer_index
        )
        self.sf = ls.sf

    def get_index_points(
        self, structures: List[pymatgen.Structure]
    ) -> List[np.ndarray]:
        """Determine and preprocess index points for GP training.

        Args:
            structures: A list of structures to convert to inputs.

        Returns:
            index_points: The feature arrays of the structures.

        """
        ls = LayerScaler(self.gnn, self.sf, self.layer_index)
        return ls.structures_to_input(structures)

    def evaluate(
        self, dataset: Literal["train", "val"], just_gnn: bool = False
    ) -> Dict[str, int]:
        """Evaluate the model on either the training or test data.

        Args:
            dataset: Which of the datasets to use.
            just_gnn: Whether to exclusively evaluate the GNN performance, or the entire model's.

        Returns:
            metrics: Names and values of metrics.

        """
        if just_gnn:
            return self.evaluate_gnn(dataset)
        else:
            return self.evaluate_uq(dataset)

    @abstractmethod
    def evaluate_gnn(self, dataset: Literal["train", "val"]) -> Dict[str, int]:
        """Evaluate the GNN's performance."""
        raise NotImplementedError()

    def evaluate_uq(self, dataset: Literal["train", "val"]) -> Dict[str, int]:
        """Evaluate the uncertainty quantifier's performance."""
        if self.training_stage < 2:
            # Not fully trained
            raise ValueError("GP not trained")

        eval_model = self.gp.model
        metric_names = eval_model.metrics_names

        if dataset == "train":
            index_points = np.stack(self.get_index_points(self.train_structs))
            targets = targets_to_tensor(self.train_targets)
        elif dataset == "val":
            index_points = np.stack(self.get_index_points(self.val_structs))
            targets = targets_to_tensor(self.val_targets)
        else:
            raise ValueError("`dataset` must be either 'train' or 'val'")

        index_points = convert_index_points(index_points)

        metric_values = eval_model.evaluate(index_points, targets)
        return {
            metric_name: metric_value
            for metric_name, metric_value in zip(metric_names, metric_values)
        }

    def train_uq(
        self, epochs: int = 500, **kwargs
    ) -> Iterator[Optional[Dict[str, float]]]:
        """Train the uncertainty quantifier.

        Extracts chosen layer outputs from :attr:`gnn`,
        scale them and train the appropriate GP (from :attr:`gp_type`).

        Yields:
            metrics: The calculated metrics at every step of training.
                (Only for `gp_type='GP'`).

        """
        training_idxs = np.stack(self.get_index_points(self.train_structs))
        val_idxs = np.stack(self.get_index_points(self.val_structs))

        training_idxs = convert_index_points(training_idxs)
        val_idxs = convert_index_points(val_idxs)

        if self.gp_type == "GP":
            yield from self._train_gp(training_idxs, val_idxs, epochs, **kwargs)
        else:
            self._train_vgp(training_idxs, val_idxs, epochs, **kwargs)
            yield None

    def _train_gp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ) -> Iterator[Dict[str, float]]:
        """Train a GP on preprocessed layer outputs from a model."""
        if self.gp_type != "GP":
            raise ValueError("Can only train GP for `gp_type='GP'`")

        train_targets = tf.constant(np.stack(self.train_targets), dtype=tf.float64)
        val_targets = tf.constant(np.stack(self.val_targets), dtype=tf.float64)

        self.gp = GPTrainer(
            train_idxs,
            train_targets,
            checkpoint_dir=str(self.gp_ckpt_path),
            kernel=self.kernel,
        )
        yield from self.gp.train_model(
            val_idxs, val_targets, epochs, save_dir=str(self.gp_save_path), **kwargs
        )

        self.kernel = self.gp.kernel

    def _train_vgp(
        self,
        train_idxs: List[np.ndarray],
        val_idxs: List[np.ndarray],
        epochs: int,
        **kwargs,
    ) -> None:
        """Train a VGP on preprocessed layer outputs from a model."""
        if self.gp_type != "VGP":
            raise ValueError("Can only train VGP for `gp_type='VGP'`")
        if self.num_inducing_points is None:
            # This should already have been handled in __init__, but just in case
            raise ValueError("Cannot train VGP without `num_inducing_points`")

        train_targets = targets_to_tensor(self.train_targets)
        val_targets = targets_to_tensor(self.val_targets)

        self.gp = SingleLayerVGP(train_idxs, self.num_inducing_points, self.ntarget)
        self.gp.train_model(
            train_targets,
            (val_idxs, val_targets),
            epochs,
            checkpoint_path=str(self.gp_ckpt_path),
            **kwargs,
        )  # type: ignore
        self.gp.model.save_weights(str(self.gp_save_path))

        self.kernel = self.gp.kernel

    def _get_vgp_dist(self, structs: List[pymatgen.Structure]):
        """Get VGP calculated distributions of target values for some structures.

        Args:
            struct: The structures to make predictions on.

        Returns:
            dist: The distributions.

        """
        if self.gp is None:
            raise ValueError(
                "UQ must be trained using `train_uq` before making predictions."
            )

        index_points = np.stack(self.get_index_points(structs))
        index_points = tf.constant(index_points, dtype=tf.float64)
        return self.gp(index_points)

    def predict_structure(
        self, struct: pymatgen.Structure
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict target value and an uncertainty for a given structure.

        Args:
            struct: The structure to make predictions on.

        Returns:
            predicted_target: The predicted target value(s).
            uncertainty: The uncertainty in the predicted value(s).

        """
        if self.gp is None:
            raise ValueError(
                "UQ must be trained using `train_uq` before making predictions."
            )

        index_point = self.get_index_points([struct])[0]
        index_point = tf.constant(index_point, dtype=tf.float64)
        predicted, uncert = self.gp.predict(index_point)
        return predicted.numpy(), uncert.numpy()

    def _validate_id_len(self, ids: Optional[List[str]], is_train: bool):
        """Check that the supplied IDs' length matches the length of saved data.

        Passes by default if `ids is None`.

        Args:
            ids: The IDs to check.
            is_train: Whether the supplied IDs correspond to train data (`True`)
                or validation data (`False`).

        Raises:
            ValueError: If there is a length mismatch.

        """
        if ids is not None:
            id_name = "train" if is_train else "val"
            struct_len = len(self.train_structs if is_train else self.val_structs)

            if (id_len := len(ids)) != struct_len:
                raise ValueError(
                    f"Length of supplied `{id_name}_materials_ids`, {id_len}, "
                    f"does not match length of `{id_name}_structs`, {struct_len}"
                )

    def save(
        self,
        train_materials_ids: Optional[List[str]] = None,
        val_materials_ids: Optional[List[str]] = None,
    ):
        """Save the full-stack model.

        Args:
            train_materials_ids: A list of IDs corresponding to :attr:`train_structs`.
                Used for indexing in the saved database.
            val_materials_ids: A list of IDs corresponding to :attr:`val_structs`.
                Used for indexing in the saved database.

        """
        for validation_args in [
            (train_materials_ids, True),
            (val_materials_ids, False),
        ]:
            self._validate_id_len(*validation_args)

        # * Write training + validation data
        if not self.train_database.exists():
            train_data = self._gen_serial_data(self.train_structs, self.train_targets)
            val_data = self._gen_serial_data(self.val_structs, self.val_targets)

            train_df = pd.DataFrame(train_data, train_materials_ids)
            val_df = pd.DataFrame(val_data, val_materials_ids)

            feather.write_feather(train_df, self.train_database)
            feather.write_feather(val_df, self.val_database)

        # * Write sf
        if self.sf is not None:
            with self.sf_path.open("wb") as f:
                np.save(f, self.sf)

        # * Write metadata
        self._write_metadata()

        # * Write kernel
        with self.kernel_save_path.open("wb") as f:
            pickle.dump(self.kernel, f)

        # * Write the GNN
        self.save_gnn()

    def _gen_serial_data(
        self, structs: List[pymatgen.Structure], targets: List[Union[float, np.ndarray]]
    ) -> Dict[str, List[Union[str, float, bytes]]]:
        """Convert a list of structures into a precursor dictionary for a DataFrame."""
        data = {"struct": [struct.to("json") for struct in structs]}

        if self.ntarget > 1:
            data["target"] = [serialize_array(arr) for arr in targets]
        else:
            data["target"] = [
                (target.item() if isinstance(target, np.ndarray) else target)
                for target in targets
            ]

        # ? Currently no need to save preprocessed index_points; the structures suffice
        # if self.training_stage > 0:
        #     data["index_points"] = [
        #         serialize_array(ips) for ips in self.get_index_points(structs)
        #     ]

        return data

    def _write_metadata(self):
        """Write metadata to a file.

        Metadata contains :attr:`gp_type`, :attr:`num_inducing_points`,
        :attr:`layer_index`, :attr:`ntarget` and :attr:`training_stage`,
        as well as the serialised :attr:`sf`.

        """
        meta = {
            "gp_type": self.gp_type,
            "num_inducing_points": self.num_inducing_points,
            "layer_index": self.layer_index,
            "ntarget": self.ntarget,
            "training_stage": self.training_stage,
        }
        with self.meta_path.open("w") as f:
            json.dump(meta, f)

    @staticmethod
    def _load_serial_data(fname: Union[Path, str]) -> pd.DataFrame:
        """Load serialized data.

        The reverse of :meth:`_gen_serial_data`.

        """
        data = feather.read_feather(fname)
        data["struct"] = data["struct"].apply(pymatgen.Structure.from_str, fmt="json")

        if isinstance(data["target"][0], bytes):
            # Data is serialized
            data["target"] = data["target"].apply(deserialize_array)

        # ? index_points not currently saved
        # try:
        #     data["index_points"] = data["index_points"].apply(deserialize_array)
        # except KeyError:
        #     # No index points in dataset
        #     pass

        return data

    @classmethod
    def load(cls, dirname: Union[Path, str]) -> ProbGNN:
        """Load a full-stack model."""
        data_dir = Path(dirname) / "data"
        train_datafile = data_dir / "train.fthr"
        val_datafile = data_dir / "val.fthr"
        kernel_save_path = Path(dirname) / "kernel.pkl"

        # * Load serialized training + validation data
        train_data = cls._load_serial_data(train_datafile)
        val_data = cls._load_serial_data(val_datafile)

        # * Load metadata
        metafile = data_dir / "meta.txt"
        with metafile.open("r") as meta_io:
            meta = json.load(meta_io)

        # * Load scaling factor, if already calculated
        sf_dir = data_dir / "sf.npy"
        sf = None
        if meta["training_stage"] > 0:
            with sf_dir.open("rb") as sf_io:  # type: ignore
                sf = np.load(sf_io)

        # * Load kernel
        with kernel_save_path.open("rb") as kernel_io:
            kernel = pickle.load(kernel_io)

        return cls(
            train_data["struct"],
            train_data["target"],
            val_data["struct"],
            val_data["target"],
            save_dir=dirname,
            sf=sf,
            kernel=kernel,
            **meta,
        )

    def change_kernel_type(
        self,
        new_kernel: Union[tfp.math.psd_kernels.PositiveSemidefiniteKernel, KernelLayer],
        new_save_dir: Path,
    ) -> ProbGNN:
        """Create a copy of the model with a different kernel type.

        The GP of the copy is overwritten.

        Args:
            new_kernel: The new kernel object.
            new_save_dir: The new saving location.

        Returns:
            The altered `ProbGNN`.

        """
        new_model = deepcopy(self)
        new_model.assign_save_directories(new_save_dir)
        new_model._mutate_kernel(new_kernel)

        if self.training_stage > 0:
            new_model.save_gnn()

        return new_model

    def _mutate_kernel(
        self,
        new_kernel: Union[tfp.math.psd_kernels.PositiveSemidefiniteKernel, KernelLayer],
    ):
        """Change the kernel type, overwriting the uncertainty quantifier.

        End users should use :meth:`change_kernel_type` so as not to
        accidentally overwrite the current model.

        Args:
            new_kernel: The new kernel

        """
        self.gp = None
        self.kernel = new_kernel
        self._validate_kernel()

    def change_gp_type(
        self,
        new_kernel: Union[tfp.math.psd_kernels.PositiveSemidefiniteKernel, KernelLayer],
        new_save_dir: Path,
        new_num_inducing_points: Optional[int] = None,
    ) -> ProbGNN:
        """Change the GP type.

        Requires the kernel to be overwritten.

        Args:
            new_kernel: The new kernel.
            new_save_dir: The new save directory.
            new_num_inducing_points: The number of inducing points. Needed if changing to VGP.

        Returns:
            The altered `ProbGNN`.

        """
        if self.gp_type != "VGP" and new_num_inducing_points is None:
            raise ValueError("Must specify number of inducing points for the VGP.")

        new_model = deepcopy(self)
        new_model.assign_save_directories(new_save_dir)
        new_model.gp_type = "GP" if self.gp_type == "VGP" else "VGP"
        new_model._mutate_kernel(new_kernel)

        if self.training_stage > 0:
            new_model.save_gnn()

        if new_model.gp_type == "VGP":
            new_model.num_inducing_points = new_num_inducing_points

        return new_model

    def change_num_inducing_points(
        self, new_num_inducing_points: int, new_save_dir: Path
    ) -> ProbGNN:
        """Change the number of VGP inducing points.

        Args:
            new_num_inducing_points: The new number of inducing points.
            new_save_dir: The save directory for the new model.

        """
        if self.gp_type != "VGP":
            raise ValueError(
                "Not using a VGP, cannot change number of inducing points."
            )

        new_model = deepcopy(self)
        new_model.assign_save_directories(new_save_dir)
        new_model.gp = None
        new_model.num_inducing_points = new_num_inducing_points

        if self.training_stage > 0:
            new_model.save_gnn()

        return new_model

    def __deepcopy__(self, memodict={}) -> ProbGNN:
        """Create a deepcopy."""
        self.save()
        return self.load(self.save_dir)


class MEGNetProbModel(ProbGNN):
    """A base MEGNetModel with uncertainty quantification.

    Args:
        train_structs: The training structures.
        train_targets: The training targets.
        val_structs: The validation structures.
        val_targets: The validation targets.
        gp_type: The method to use for the Gaussian process.
            Must be either 'GP' or 'VGP'.
        save_dir: The directory to save files to during training.
            Files include MEGNet and GP checkpoints.
        ntarget: The number of target variables.
            This can only be greater than one if `gp_type` is 'VGP'.
        layer_index: The index of the layer to extract outputs from
            within :attr:`gnn`. Defaults to the concatenation
            layer.
        num_inducing_points: The number of inducing points for the `VGP`.
            Can only be set for `gp_type='VGP'`.
        training_stage: The stage of training the model is at.
            Only applies when loading a model.
        sf: The pre-calculated scaling factor. Only applicable when loading
            a pre-trained model.
        **kwargs: Keyword arguments to pass to :class:`MEGNetModel`.

    """

    def make_gnn(self, metrics=["MeanAbsoluteError"], **kwargs) -> MEGNetModel:
        """Create a new MEGNetModel."""
        try:
            meg_model = MEGNetModel(ntarget=self.ntarget, metrics=metrics, **kwargs)
        except ValueError:
            meg_model = MEGNetModel(
                ntarget=self.ntarget,
                metrics=metrics,
                **kwargs,
                **get_default_megnet_args(),
            )

        try:
            meg_model.model.load_weights(str(self.gnn_ckpt_path))
        except NotFoundError:
            pass

        return meg_model

    def load_gnn(self) -> MEGNetModel:
        """Load a saved MEGNetModel."""
        return MEGNetModel.from_file(str(self.gnn_save_path))

    def train_gnn(
        self,
        epochs: Optional[int] = 1000,
        batch_size: Optional[int] = 128,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        **kwargs,
    ):
        """Train the MEGNetModel.

        Args:
            epochs: The number of training epochs.
            batch_size: The batch size.
            callbacks: Callbacks to use during training.
                Will always add a checkpoint callback.
            **kwargs: Keyword arguments to pass to :func:`MEGNetModel.train`.

        """
        checkpoint_file_path = (
            self.gnn_ckpt_path
            / "val_mae_{epoch:05d}_{val_mean_absolute_error:.6f}.hdf5"
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_file_path),
            monitor="val_mean_absolute_error",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(checkpoint_callback)

        self.gnn.train(
            self.train_structs,
            self.train_targets,
            self.val_structs,
            self.val_targets,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            save_checkpoint=False,
            dirname=str(self.gnn_ckpt_path),
            **kwargs,
        )

        self.save_gnn()
        self._update_sf()

    def evaluate_gnn(self, dataset: Literal["train", "val"]) -> Dict[str, int]:
        """Evaluate the MEGNet model's performance."""
        if dataset == "train":
            structs = self.train_structs
            targets = np.stack(self.train_targets)
        elif dataset == "val":
            structs = self.val_structs
            targets = np.stack(self.val_targets)
        else:
            raise ValueError("`dataset` must be either 'train' or 'val'")

        predicted = self.gnn.predict_structures(structs)

        return {
            "mae": mean_absolute_error(targets, predicted),
            "mape": mean_absolute_percentage_error(
                np.stack(targets), np.stack(predicted)
            ),
        }

    def save_gnn(self):
        self.gnn.save_model(str(self.gnn_save_path))


def targets_to_tensor(targets: List[Union[float, np.ndarray]]) -> tf.Tensor:
    """Convert a list of target values to a Tensor."""
    return tf.constant(np.stack(targets), dtype=tf.float64)


def get_default_megnet_args(
    nfeat_bond: int = 10, r_cutoff: float = 5.0, gaussian_width: float = 0.5
) -> dict:
    """Get default MEGNet arguments.

    These are the fallback for when no graph converter is supplied,
    taken from the MEGNet Github page.

    Args:
        nfeat_bond: Number of bond features. Default (10) is very low, useful for testing.
        r_cutoff: The atomic radius cutoff, above which to ignore bonds.
        gaussian_width: The width of the gaussian to use in determining bond features.

    Returns:
        megnet_args: Some default-ish MEGNet arguments.

    """
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    graph_converter = CrystalGraph(cutoff=r_cutoff)
    return {
        "graph_converter": graph_converter,
        "centers": gaussian_centers,
        "width": gaussian_width,
    }
