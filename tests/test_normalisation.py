"""Check the conditioning contract and prevent validation-fitted scaling."""
import os
os.environ.setdefault('DATA_ROOT', '/tmp/unused-wwdc-test-data')
import unittest
from unittest.mock import patch
import numpy as np
import torch
from wwdc_spectra.data.sdss import SDSS
from wwdc_spectra.data.normalisation import configure_normalisation, fit_log_scaling


class Rows(list):
    def select(self, indices):
        return Rows(self[i] for i in indices)


class NormalisationTests(unittest.TestCase):
    def source(self, variables):
        source = SDSS.__new__(SDSS)
        source.dataset = Rows([{'spectrum': dict(flux=[], ivar=[], **{'lambda': []}, mask=[]), 'Z': 0.2}])
        source.y_catalog = {'variables': variables, 'drop_variables': []}
        source.return_id = source.return_pos = source.return_mask_ratio = source.return_mask = False
        source.normalisation_stats = {'mean': 1.0, 'std': 0.5}
        source.prepare_spectrum = lambda **kw: (torch.ones(3), torch.ones(3), torch.tensor(100.), torch.ones(3), kw['z'])
        return source

    def test_norm_is_preprocessing_factor_not_catalogue_or_redshift(self):
        source = self.source({'NORM': {}})
        spec, y = source[0]
        torch.testing.assert_close(y, torch.tensor([2.]))
        self.assertEqual(spec.shape, (3,))
        self.assertEqual(y.dtype, torch.float32)

    def test_existing_redshift_condition_is_unchanged(self):
        source = self.source({'Z': {}})
        torch.testing.assert_close(source[0][1], torch.tensor([0.2]))

    def test_invalid_factor_is_not_silently_logged(self):
        source = self.source({'NORM': {}})
        for value in [0., -1., float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                source.normalisation_condition(value)

    def test_statistics_and_validation_membership(self):
        mean, std = fit_log_scaling(np.array([1., 10., 100., 0., -3., np.nan]))
        self.assertAlmostEqual(mean, 1.)
        self.assertAlmostEqual(std, np.std([0., 1., 2.]))
        source = self.source({'NORM': {}})
        source.dataset = Rows(['a', 'b', 'c'])
        # Validation values are deliberately far from training values.
        with patch('wwdc_spectra.data.sdss.SDSS') as training, patch('wwdc_spectra.data.normalisation.cached_norms', side_effect=[(np.array([1., 10., 100.]), 'train.npy'), (np.array([1000., 0., 10000.]), 'val.npy')]):
            configure_normalisation(source, 'val', '/tmp/unused', 0)
        self.assertEqual(source.dataset, ['a', 'c'])
        self.assertEqual(source.normalisation_stats['mean'], mean)
        self.assertEqual(source.normalisation_stats['std'], std)
        self.assertEqual(source.normalisation_stats['excluded_rows'], 1)
        self.assertEqual(source.normalisation_stats['fit_split'], 'train')


if __name__ == '__main__':
    unittest.main()
