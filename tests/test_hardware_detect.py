"""Détection matérielle : ce qu'on lit du pilote, et ce qu'on en déduit.

Le point de ces tests : l'architecture d'une carte se déduit de sa CAPACITÉ DE
CALCUL, pas de son nom commercial. Le nom reste un repli pour les pilotes trop
anciens pour l'exposer — et le repli doit continuer de marcher, sinon une vieille
machine ne démarre plus.
"""
import unittest
from unittest.mock import patch

from atelier import hardware as h


class ArchFromComputeCapTests(unittest.TestCase):
    def test_known_capabilities(self):
        self.assertEqual(h._arch_from_cc("6.1"), ("pascal", False))
        self.assertEqual(h._arch_from_cc("7.5"), ("turing", True))
        self.assertEqual(h._arch_from_cc("8.6"), ("ampere", True))
        self.assertEqual(h._arch_from_cc("8.9"), ("ada", True))
        self.assertEqual(h._arch_from_cc("12.0"), ("blackwell", True))

    def test_future_capability_is_not_downgraded(self):
        # Une puce plus récente que notre table ne doit pas hériter d'un profil
        # dégradé : depuis Volta, NVIDIA n'a jamais retiré les tensor cores.
        arch, tc = h._arch_from_cc("13.5")
        self.assertTrue(tc)

    def test_unreadable_capability(self):
        for bad in ("", "bogus", None):
            self.assertIsNone(h._arch_from_cc(bad))


class GpuLabelTests(unittest.TestCase):
    def test_sm_comes_from_the_driver_when_available(self):
        g = h.Gpu(0, "RTX 2080 Ti", 11.0, "turing", True, "7.5", "550")
        self.assertEqual(g.sm, "sm_75")

    def test_sm_falls_back_to_architecture(self):
        g = h.Gpu(0, "RTX 3060", 12.0, "ampere", True, "", "")
        self.assertEqual(g.sm, "sm_86")

    def test_label_is_self_contained(self):
        g = h.Gpu(0, "RTX 2080 Ti", 11.0, "turing", True, "7.5", "550")
        for bit in ("RTX 2080 Ti", "11 GB", "turing", "sm_75"):
            self.assertIn(bit, g.label())


class DetectGpusTests(unittest.TestCase):
    MODERN = "0, NVIDIA GeForce RTX 3060, 12288, 8.6, 550.54\n"
    OLD = "0, NVIDIA GeForce RTX 3060, 12288\n"

    def _detect(self, responses):
        """responses : dict champs -> sortie simulée (None = requête refusée)."""
        h.detect_gpus.cache_clear()
        h._apple_gpu.cache_clear()
        with patch.object(h, "_apple_gpu", return_value=None), \
             patch.object(h, "_nvidia_smi",
                          side_effect=lambda f: responses.get(f)):
            return h.detect_gpus()

    def tearDown(self):
        h.detect_gpus.cache_clear()
        h._apple_gpu.cache_clear()

    def test_modern_driver_gives_compute_cap(self):
        gpus = self._detect({
            "index,name,memory.total,compute_cap,driver_version": self.MODERN})
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].compute_cap, "8.6")
        self.assertEqual(gpus[0].arch, "ampere")
        self.assertEqual(gpus[0].driver, "550.54")

    def test_old_driver_still_detects_the_card(self):
        # Le pilote refuse la requête complète : on doit retomber sur la courte
        # plutôt que de conclure « aucun GPU » et basculer en mode CPU.
        gpus = self._detect({"index,name,memory.total": self.OLD})
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].arch, "ampere")   # déduit du nom
        self.assertEqual(gpus[0].compute_cap, "")

    def test_gtx_16xx_keeps_its_lack_of_tensor_cores(self):
        # 7.5 comme une RTX 20xx, mais sans tensor cores : seul le nom permet
        # de les distinguer, et flash-attention en dépend.
        gpus = self._detect({
            "index,name,memory.total,compute_cap,driver_version":
                "0, NVIDIA GeForce GTX 1660 SUPER, 6144, 7.5, 550.54\n"})
        self.assertEqual(gpus[0].arch, "turing")
        self.assertFalse(gpus[0].tensor_cores)

    def test_no_driver_at_all(self):
        self.assertEqual(self._detect({}), ())


class FlashAttentionFollowsTensorCoresTests(unittest.TestCase):
    def _profile(self, gpu):
        h.detect_gpus.cache_clear()
        with patch.object(h, "detect_gpus", return_value=(gpu,)), \
             patch.object(h, "detect_ram_gb", return_value=32.0):
            return h.auto_profile(None)

    def tearDown(self):
        h.detect_gpus.cache_clear()

    def test_gtx_1660_does_not_get_flash_attention(self):
        gpu = h.Gpu(0, "NVIDIA GeForce GTX 1660 SUPER", 6.0, "turing", False,
                    "7.5", "550")
        self.assertFalse(self._profile(gpu).diffusion_fa)

    def test_rtx_2080ti_does(self):
        gpu = h.Gpu(0, "NVIDIA GeForce RTX 2080 Ti", 11.0, "turing", True,
                    "7.5", "550")
        self.assertTrue(self._profile(gpu).diffusion_fa)


class FreeVramTests(unittest.TestCase):
    def _free(self, out, index=None):
        with patch.object(h, "_apple_gpu", return_value=None), \
             patch.object(h, "_nvidia_smi", return_value=out):
            return h.free_vram_gb(index)

    def test_reads_the_requested_card(self):
        self.assertAlmostEqual(
            self._free("0, 2048\n1, 10240\n", index=1), 10.0, places=1)

    def test_without_index_takes_the_roomiest(self):
        self.assertAlmostEqual(self._free("0, 2048\n1, 10240\n"), 10.0, places=1)

    def test_unavailable_is_zero_not_a_guess(self):
        self.assertEqual(self._free(None), 0.0)


if __name__ == "__main__":
    unittest.main()
