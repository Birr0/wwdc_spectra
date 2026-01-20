import os

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from dotenv import load_dotenv
from torch.utils.data import Dataset
from spender.instrument import get_skyline_mask

load_dotenv()
RAW_DATA_DIR = os.getenv("DATA_ROOT") + "/sdss/"
DATA_DIR = os.getenv("DATA_ROOT") + "/sdss/sdss_II_catalog"
LOADING_SCRIPT = "./sdss_II_resources/sdss_mmu.py"


class SDSS(Dataset):
    def __init__(
        self, 
        split="train", 
        x_ds=None, 
        y_catalog=None, 
        return_id=False, 
        return_pos=False
    ):
        if not self.data_exists():
            if not self.raw_data_exists():
                msg = f"""
                Data not found in {RAW_DATA_DIR}. Download this using instructions from \
                https://github.com/MultimodalUniverse/MultimodalUniverse/tree/main/scripts/sdss
                """
                raise ValueError(msg)
            else:
                dataset = load_dataset(
                    LOADING_SCRIPT,
                    data_dir=RAW_DATA_DIR,
                    trust_remote_code=True,
                    cache_dir=RAW_DATA_DIR,
                )

                dataset_traintest = dataset["train"].train_test_split(
                    test_size=0.2, seed=42
                )  # Split the data to 0.8:02

                dataset_valtest = dataset_traintest["test"].train_test_split(
                    test_size=0.1, seed=42
                )  # Split the first split in half from validation and test.

                dataset = DatasetDict(
                    {
                        "train": dataset_traintest["train"],
                        "val": dataset_valtest["train"],
                        "test": dataset_valtest["test"],
                    }
                )
                # Need to merge the catalog with the dataset.
                os.mkdir(DATA_DIR)
                dataset.save_to_disk(DATA_DIR)

        if split not in ["train", "test", "val"]:
            msg = f"Invalid split: {split}. \
            Must be 'train' or 'test' or 'val'."
            raise ValueError(msg)

        self.x_ds = x_ds
        self.y_catalog = y_catalog

        self.dataset = load_from_disk(DATA_DIR)[split]
        self._wave_obs = 10 ** torch.arange(3.578, 3.97, 0.0001)
        self._skyline_mask = get_skyline_mask(self._wave_obs)      
        self.return_id = return_id  
        self.return_pos = return_pos
    
    @staticmethod
    def raw_data_exists():
        # simple check of the path.
        return os.path.exists(RAW_DATA_DIR)

    @staticmethod
    def data_exists():
        # simple check of the path.
        return os.path.exists(DATA_DIR)

    def prepare_spectrum(
        self,
        flux,
        ivar,
        wavelengths,
        mask,
        z,
    ):
        """Prepare spectrum for analysis

        This method creates an extended mask, using the original SDSS `and_mask` and
        the skyline mask of the instrument. The weights for all masked regions is set to
        0, but the spectrum itself is not altered.

        The spectrum and weights are then cast into a fixed-format data vector, which
        standardizes the variable wavelengths of each observation.

        A normalization is computed as the median flux in the relatively flat region
        between restframe 5300 and 5850 A. The spectrum is divided by this factor, the weights
        are muliplied with the square of this factor.

        Parameter
        ---------
        flux: float
            Spectra flux
        ivar: float
            Invariance variance of spectrum
        wavelengths: float
            Wavelengths for each flux measurement
        mask: bool
            Masks for flux measurements
        z: float
            Redshift of the spectrum
        z_err: float
            Uncertainity in redshift.


        Returns
        -------
        spec: `torch.tensor`, shape (L, )
            Normalized spectrum
        w: `torch.tensor`, shape (L, )
            Inverse variance weights of normalized spectrum
        norm: `torch.tensor`, shape (1, )
            Flux normalization factor
        z: `torch.tensor`, shape (1, )
            Redshift (only returned when argument z=None)
        zerr: `torch.tensor`, shape (1, )
            Redshift error (only returned when argument z=None)
        """
        loglam = torch.log10(
            torch.tensor(wavelengths)
        )
        flux = torch.tensor(flux, dtype=torch.float32)  #np.array(obj["spectrum"]["flux"])

        flux_nan = flux.isnan()

        ivar = torch.tensor(ivar, dtype=torch.float32) #np.array(obj["spectrum"]["ivar"])

        ivar[mask] = 0

        # loglam is subset of _wave_obs, need to insert into extended tensor
        L = len(self._wave_obs)
        start = int(
            torch.round((loglam[0] - torch.log10(self._wave_obs[0]).item()) / 0.0001)
        )
        if start < 0:
            flux = flux[-start:]
            ivar = ivar[-start:]
            end = min(start + len(loglam), L)
            start = 0
        else:
            end = min(start + len(loglam), L)
        spec = torch.zeros(L)
        w = torch.zeros(L)

        spec[start:end] = flux
        w[start:end] = ivar

        # remove regions around skylines
        w[self._skyline_mask] = 0

        # normalize spectrum:
        # for redshift invariant encoder: select norm window in restframe
        wave_rest = self._wave_obs / (1 + z)
        # flatish region that is well observed out to z ~ 0.5
        sel = (w > 0) & (wave_rest > 5300) & (wave_rest < 5850)
        if sel.count_nonzero() == 0:
            norm = torch.tensor(0)
        else:
            norm = torch.median(spec[sel])
        # remove spectra (from training) for which no valid norm could be found
        if not torch.isfinite(norm):
            norm = 0
        else:
            spec /= norm
        w *= norm**2

        # This is required to remove NaNs from the spectrum, if present.
        # Otherwise, the loss is poisioned.
        finite_spec = torch.isfinite(spec)
        finite_w = torch.isfinite(w)
        good = finite_spec & finite_w
        # zero spectrum where non-finite; zero weights where either is bad
        spec = torch.where(finite_spec, spec, torch.zeros_like(spec))
        w = torch.where(good, w, torch.zeros_like(w))

        return spec, w, norm, wave_rest, z

    @staticmethod
    def _to_f32_or_nan(x):
        return torch.tensor(
            float(x) if x is not None else float("nan"), dtype=torch.float32
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # pre-process spectrum
        spec, w, _, wave_rest, z = self.prepare_spectrum(
            flux=item["spectrum"]["flux"],
            ivar=item["spectrum"]["ivar"],
            wavelengths=item["spectrum"]["lambda"],
            mask=item["spectrum"]["mask"],
            z=item["Z"],
        )

        # Pre-process conditioning data
        y = torch.stack(
            [
                torch.as_tensor(item[k])
                for k in self.y_catalog.variables
                if k not in self.y_catalog["drop_variables"]
            ]
        )

        if self.return_id and (not self.return_pos):
            return spec, y, item["BESTOBJID"]
        elif (not self.return_id) and self.return_pos:
            return spec, y, item["ra"], item["dec"], item["Z"]
        elif self.return_id and self.return_pos:
            return spec, y, item["BESTOBJID"], item["ra"], item["dec"], item["Z"]
        return spec, y

class SDSS_AION(SDSS):
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Pre-process conditioning data
        y = torch.stack(
            [
                torch.as_tensor(item[k])
                for k in self.y_catalog.variables
                if k not in self.y_catalog["drop_variables"]
            ]
        )

        return (
            item["spectrum"]["flux"],
            item["spectrum"]["ivar"],
            item["spectrum"]["lambda"],
            item["spectrum"]["mask"],
            y,
        )

        # Add a flag to return catalog. This should not happen during training!


if __name__ == "__main__":
    from modules import WWDCDataLoader

    data = SDSS_AION(split="test")
    dataset = WWDCDataLoader(data)
    for i in range(10):
        print(data[i])
        print("---")

    # print(data[0]["X"].max())
    # print(data[0]["X"].min())
    # print(data[0]["catalog"].keys())

    # check the lightning loader here.
    # Why isn't pre-processing not giving the
    # same length spectra?
