import os

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from dotenv import load_dotenv
from torch.utils.data import Dataset
from spender.instrument import get_skyline_mask
from spender.util import interp1d

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
        return_pos=False,
        return_mask_ratio=False
    ):
        if not self.data_exists(): # this needs replaced.
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
        self.return_mask_ratio = return_mask_ratio
    
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
            z=item["Z"], # Changed from original
        )

        # Pre-process conditioning data
        y = torch.stack(
            [
                torch.as_tensor(item[k])
                for k in self.y_catalog["variables"]
                if k not in self.y_catalog["drop_variables"]
            ]
        )

        out = [spec, y]
        if self.return_id:
            #objid = item.get("BESTOBJID")
            specid = item.get("object_id")
            out.append(specid if specid is not None else "")

        if self.return_pos:
            ra = item.get("ra"); dec = item.get("dec"); z = item.get("Z")
            out.extend([
                float(ra) if ra is not None else float("nan"),
                float(dec) if dec is not None else float("nan"),
                float(z) if z is not None else float("nan"),
            ])

        if self.return_mask_ratio:       
            out.append(
                (w == 0).sum()/len(w)
            )

        return tuple(out)

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

class SDSS_VOLUME_LIMITED(SDSS):
    def __init__(
        self,
        z_lim,
        split="train", 
        x_ds=None, 
        y_catalog=None, 
        return_id=False, 
        return_pos=False,
        return_mask_ratio=False,
        augment=False
    ):
        # add loader from HF here at later date.
        self.x_ds = x_ds
        self.y_catalog = y_catalog

        DATA_ROOT = os.getenv("DATA_ROOT")
        
        self.dataset = load_from_disk(
            f"{DATA_ROOT}/sdss/volume_limited/z={z_lim:.3f}"
        )[split]
        
        self._wave_obs = 10 ** torch.arange(3.578, 3.97, 0.0001)
        self._skyline_mask = get_skyline_mask(self._wave_obs)      
        self.return_id = return_id  
        self.return_pos = return_pos
        self.return_mask_ratio = return_mask_ratio
        self.z_lim = torch.tensor(z_lim)
        self.augment = augment
        print(f"Augmentations activate: {self.augment}")
    
    def augment_spectra(
        self, 
        batch, 
        redshift=True, 
        noise=True, 
        mask=True, 
        ratio=0.05, 
        z_new=None
    ):
        """Augment spectra for greater diversity

        Parameters
        ----------
        batch: `torch.tensor`, shape (N, L)
            Spectrum batch
        redshift: bool
            Modify redshift by up to 0.2 (keeping it within 0...0.5)
        noise: bool
            Whether to add noise to the spectrum (up to 0.2*max(spectrum))
        mask: bool
            Whether to block out a fraction (given by `ratio`) of the spectrum
        ratio: float
            Fraction of the spectrum that will be masked
        z_new: float
            Adopt this redshift for all spectra in the batch

        Returns
        -------
        spec: `torch.tensor`, shape (N, L)
            Altered spectrum
        w: `torch.tensor`, shape (N, L)
            Altered inverse variance weights of spectrum
        z: `torch.tensor`, shape (N, )
            Altered redshift
        """

        spec, w, z = batch[:3]
        wave_obs = self._wave_obs

        if redshift:
            if z_new == None:
                # uniform distribution of redshift offsets, width = z_lim
                z_base = torch.relu(z-self.z_lim)
                z_new = z_base+self.z_lim*(torch.rand(1))
            # keep redshifts between 0 and z_lim
            z_new = torch.minimum(
                torch.nn.functional.relu(z_new),
                0.5 * torch.ones(1),
            )
            zfactor = (1 + z_new) / (1 + z)
            wave_redshifted = (wave_obs * zfactor).T

            # redshift linear interpolation
            spec_new = interp1d(wave_redshifted, spec, wave_obs)
            # ensure extrapolated values have zero weights
            w_new = torch.clone(w)
            w_new[0] = 0
            w_new[-1] = 0
            w_new = interp1d(wave_redshifted, w_new, wave_obs)
            w_new = torch.nn.functional.relu(w_new)
        else:
            spec_new, w_new, z_new = torch.clone(spec), torch.clone(w), z

        # add noise
        if noise:
            sigma = 0.2 * torch.max(spec, 0, keepdim=True)[0]
            noise = sigma * torch.distributions.Normal(0, 1).sample()
            noise_mask = (
                torch.distributions.Uniform(0, 1).sample() > ratio
            )
            noise[noise_mask] = 0
            spec_new += noise
            # add variance in quadrature, avoid division by 0
            w_new = 1 / (1 / (w_new + 1e-6) + noise**2)

        if mask:
            length = int(len(spec) * ratio)
            start = torch.randint(0, len(spec) - length, (1,)).item()
            spec_new[start : start + length] = 0
            w_new[start : start + length] = 0

        return spec_new, w_new, z_new
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # pre-process spectrum
        spec, w, _, wave_rest, z = self.prepare_spectrum(
            flux=item["spectrum"]["flux"],
            ivar=item["spectrum"]["ivar"],
            wavelengths=item["spectrum"]["lambda"],
            mask=item["spectrum"]["mask"],
            z=item["Z_raw"],
        )

        if self.augment:
            spec, w, z = self.augment_spectra(
                (spec, w, z), 
                self.z_lim, 
                redshift=True, 
                noise=True, 
                mask=True, 
                ratio=0.05, 
                z_new=None
            )
            
            spec = spec.squeeze(0)
            w = w.squeeze(0)

        # Pre-process conditioning data
        y = torch.stack(
            [
                torch.as_tensor(item[k])
                for k in self.y_catalog.variables
                if k not in self.y_catalog["drop_variables"]
            ]
        )

        out = [spec, y]
        if self.return_id:
            objid = item.get("BESTOBJID")
            out.append(objid if objid is not None else "")
            #vacid = item.get("VAC_ID")
            #out.append(vacid if vacid is not None else "")

        if self.return_pos:
            ra = item.get("ra"); dec = item.get("dec"); z = item.get("Z_raw")
            out.extend([
                float(ra) if ra is not None else float("nan"),
                float(dec) if dec is not None else float("nan"),
                float(z) if z is not None else float("nan"),
            ])

        if self.return_mask_ratio:       
            out.append(
                (w == 0).sum()/len(w)
            )

        return tuple(out)

class SDSS_MAGNITUDE_LIMITED(SDSS_VOLUME_LIMITED):
    def __init__(
        self,
        z_lim,
        split="train", 
        x_ds=None, 
        y_catalog=None, 
        return_id=False, 
        return_pos=False,
        return_mask_ratio=False,
        augment=False
    ):
        # add loader from HF here at later date.
        self.x_ds = x_ds
        self.y_catalog = y_catalog

        DATA_ROOT = os.getenv("DATA_ROOT")

        self.dataset = load_from_disk(
            f"{DATA_ROOT}/sdss/magnitude_limited/z={z_lim:.3f}"
        )[split]
        self._wave_obs = 10 ** torch.arange(3.578, 3.97, 0.0001)
        self._skyline_mask = get_skyline_mask(self._wave_obs)      
        self.return_id = return_id  
        self.return_pos = return_pos
        self.return_mask_ratio = return_mask_ratio
        self.z_lim = torch.tensor(z_lim)
        self.augment = augment
        print(f"Augmentations activate: {self.augment}")

class SDSS_MAGNITUDE_SLICE(SDSS_VOLUME_LIMITED):
    def __init__(
        self,
        z_lim,
        mag_lim,
        split="train", 
        x_ds=None, 
        y_catalog=None, 
        return_id=False, 
        return_pos=False,
        return_mask_ratio=False,
        augment=False
    ):
        # add loader from HF here at later date.
        self.x_ds = x_ds
        self.y_catalog = y_catalog

        DATA_ROOT = os.getenv("DATA_ROOT")

        self.dataset = load_from_disk(
            f"{DATA_ROOT}/sdss/mag_slices/z={z_lim:.2f}/M_max={mag_lim:.1f}_spectra/"
        )[split]
        self._wave_obs = 10 ** torch.arange(3.578, 3.97, 0.0001)
        self._skyline_mask = get_skyline_mask(self._wave_obs)      
        self.return_id = return_id  
        self.return_pos = return_pos
        self.return_mask_ratio = return_mask_ratio
        self.z_lim = torch.tensor(z_lim)
        self.augment = augment
        print(f"Augmentations activate: {self.augment}")

if __name__ == "__main__":
    from modules import WWDCDataLoader

    #train_data = #SDSS_VOLUME_LIMITED(split="train", z_lim=0.075)
    #val_data = #SDSS_VOLUME_LIMITED(split="val", z_lim=0.075)
    x_ds = {
        "dataset_name": "sdss",
        "fp": "${meta.data_path}",
        "type": "spectra",
        "augmentations": None,  # originally: none
        "size": 3921,
    }

    y_catalog = {
        "catalog_name": "sdss",
        "fp": None,
        "join_method": None,
        "variables": {
            "Z": {
                "name": "redshift",
                "size": 1,
                "processing_fn": None,
            }
        },
        "drop_variables": [],
    }

    test_data = SDSS_VOLUME_LIMITED(
        split="test",
        return_id=True,
        x_ds=x_ds, 
        y_catalog=y_catalog, 
        return_pos=True,
        return_mask_ratio=True,
        z_lim=0.200,
    )

    print(test_data[0])
    #dataset = WWDCDataLoader(train_data)
    # print(data[0]["X"].max())
    # print(data[0]["X"].min())
    # print(data[0]["catalog"].keys())
    #print(dataset)

    # check the lightning loader here.
    # Why isn't pre-processing not giving the
    # same length spectra?
