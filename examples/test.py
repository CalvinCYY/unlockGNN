"""Quick implementation test for experimental unlockGNN mode."""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymatgen
import tensorflow.keras as keras
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from unlockgnn.experimental.model import MEGNetProbModel
from megnet.models import MEGNetModel
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).parent
PROB_MODEL_PATH = THIS_DIR / "prob_model"
PROB_MODEL_IMP_PATH = THIS_DIR / "prob_model_norm"
DATA_PATH = THIS_DIR / "data.pkl"
TB_LOG_DIR = THIS_DIR / "tb_logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
TB_IMP_LOG_DIR = THIS_DIR / "tb_imp_logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
MP_API_KEY = <Insert Materials Project API key here>


def load_megnet_model() -> MEGNetModel:
    """Load MEGNet's builtin formation energies model."""
    return MEGNetModel.from_mvl_models("Eform_MP_2019")


def creation_routine(
    meg_model: MEGNetModel, use_normalization: bool
) -> MEGNetProbModel:
    """Create a new probabilistic model."""
    return MEGNetProbModel(
        num_inducing_points=500,
        save_path=PROB_MODEL_IMP_PATH if use_normalization else PROB_MODEL_PATH,
        meg_model=meg_model,
        use_normalization=use_normalization,
    )


def get_data() -> pd.DataFrame:
    """Get materials project formation energies dataframe."""
    if not DATA_PATH.exists():
        full_df = MPDataRetrieval(MP_API_KEY).get_dataframe(
            criteria={"nelements": 2, "e_above_hull": {"$eq": 0}},
            properties=["structure", "formation_energy_per_atom"],
        )
        full_df.to_pickle(DATA_PATH)
        return full_df
    else:
        return pd.read_pickle(DATA_PATH)


def train_test_routine(
    prob_model: MEGNetProbModel, example_data: pd.DataFrame, use_normalization: bool
):
    """Test training the VGP component of the model."""
    train_data, val_data = train_test_split(example_data, random_state=2021)

    train_structs = train_data["structure"]
    val_structs = val_data["structure"]
    train_targets = train_data["formation_energy_per_atom"]
    val_targets = val_data["formation_energy_per_atom"]

    tb_dir = TB_IMP_LOG_DIR if use_normalization else TB_LOG_DIR
    tensorboard_callback = keras.callbacks.TensorBoard(
        tb_dir, write_graph=False, profile_batch=0
    )

    prob_model.train_vgp(
        train_structs,
        train_targets,
        5,
        val_structs,
        val_targets,
        [tensorboard_callback],
        verbose=1,
    )


def format_entries(struct_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Add predictions to database of structures."""
    struct_df["predicted_e_form"] = predictions[:, 0]
    struct_df["uncertainty"] = predictions[:, 1] * 3
    return struct_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--norm",
        action="store_true",
        dest="norm",
        help="Set to use a normalization layer before the VGP.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        dest="eval",
        help="Set to just evaluate the model on an example input, without training.",
    )
    args = parser.parse_args()
    norm = args.norm
    evaluate = args.eval

    model_path = PROB_MODEL_IMP_PATH if norm else PROB_MODEL_PATH
    meg_model = None
    if not model_path.exists():
        meg_model = load_megnet_model()
        print("Creating model...")
        prob_model = creation_routine(meg_model, norm)
    else:
        print("Loading model...")
        prob_model = MEGNetProbModel.load(model_path)

    print("Getting data...")
    df = get_data()

    if evaluate:
        example_entry = df.iloc[0, :]
        eg_struct: pymatgen.Structure = example_entry["structure"]
        eg_energy = example_entry["formation_energy_per_atom"]

        print(f"{eg_struct.composition}: {eg_energy:.3f} eV / atom.")
        pred, stddev = prob_model.predict(eg_struct)

        print(f"Predicted: {pred:.3f} ± {stddev * 3:.3f}")
    else:
        print("Training model...")
        train_test_routine(prob_model, df, norm)

        prob_model.save()
