"""« Haute résolution » : la méthode tient en trois détails, on les verrouille.

Chacun est facile à casser sans s'en apercevoir, et chacun change le résultat :
le mot du prompt, le fait que la MÊME image parte en référence ET en départ, et
le pré-agrandissement bilinéaire. Un test qui ne vérifierait que « ça produit
une image » les laisserait tous passer.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from atelier.engine import highres, sdcpp


class SizePlanTests(unittest.TestCase):
    def test_it_never_goes_below_the_native_resolution_of_flux(self):
        # 512² agrandi ×1 ferait 0,26 Mpx : le modèle repasserait de toute
        # façon par son régime natif, autant y aller franchement.
        w, h = highres.plan_size((512, 512), 1.0)
        self.assertGreaterEqual(w * h, highres.FLUX_PIXELS * 0.98)

    def test_it_never_goes_past_what_the_model_holds_together(self):
        w, h = highres.plan_size((2048, 2048), 3.0)
        self.assertLessEqual(w * h, highres.MAX_PIXELS)

    def test_the_aspect_ratio_survives(self):
        w, h = highres.plan_size((1200, 800), 2.0)
        self.assertAlmostEqual(w / h, 1200 / 800, places=1)

    def test_sizes_are_multiples_of_sixteen(self):
        for size in ((640, 480), (1200, 800), (999, 333)):
            w, h = highres.plan_size(size, 2.0)
            self.assertEqual((w % 16, h % 16), (0, 0), size)


class ColorMatchTests(unittest.TestCase):
    """Les couleurs de l'original, le détail du résultat."""

    def test_it_brings_back_the_original_colour(self):
        source = Image.new("RGB", (64, 64), (200, 60, 40))     # rouge franc
        washed = Image.new("RGB", (64, 64), (150, 140, 135))   # fade
        fixed = highres.color_match(washed, source)
        r, g, b = fixed.getpixel((32, 32))
        self.assertGreater(r, 180)
        self.assertLess(g, 90)

    def test_it_keeps_the_detail_the_model_added(self):
        source = Image.new("RGB", (64, 64), (120, 120, 120))
        detailed = Image.new("RGB", (64, 64), (120, 120, 120))
        for x in range(0, 64, 2):                      # rayures fines
            for y in range(64):
                detailed.putpixel((x, y), (200, 200, 200))
        fixed = highres.color_match(detailed, source)
        pixels = [fixed.getpixel((x, 32))[0] for x in range(64)]
        self.assertGreater(max(pixels) - min(pixels), 40,
                           "les hautes fréquences ont été écrasées")

    def test_a_smaller_original_is_scaled_to_the_result(self):
        source = Image.new("RGB", (32, 32), (10, 200, 10))
        result = Image.new("RGB", (128, 128), (128, 128, 128))
        self.assertEqual(highres.color_match(result, source).size, (128, 128))


class PipelineTests(unittest.TestCase):
    """Ce qui part réellement au moteur."""

    def _run(self, **kwargs):
        seen = {}

        def fake_generate(**call):
            seen.update(call)
            out = Path(tempfile.mkdtemp()) / "out.png"
            Image.new("RGB", (call["width"], call["height"]), "navy").save(out)
            return [out]

        with patch.object(highres.gen_engine, "generate",
                          side_effect=fake_generate):
            result = highres.high_resolution(
                Image.new("RGB", (512, 384), (180, 90, 60)),
                model_id="flux2-klein-9b", **kwargs)
        return seen, result

    def test_the_same_image_is_reference_and_starting_point(self):
        # C'est LE cœur de la méthode : l'interface envoyait l'un ou l'autre,
        # jamais les deux, et le moteur n'a jamais eu le problème.
        seen, _ = self._run()
        self.assertEqual(seen["init_image"], seen["ref_image"])
        self.assertTrue(Path(seen["init_image"]).is_file())

    def test_the_prompt_says_high_resolution_not_upscale(self):
        seen, _ = self._run()
        self.assertIn("high resolution", seen["prompt"].lower())
        self.assertNotIn("upscale", seen["prompt"].lower())

    def test_a_custom_prompt_replaces_the_default(self):
        seen, _ = self._run(prompt="portrait, peau nette")
        self.assertEqual(seen["prompt"], "portrait, peau nette")

    def test_the_denoise_is_passed_as_strength(self):
        seen, _ = self._run(strength=0.85)
        self.assertAlmostEqual(seen["strength"], 0.85)

    def test_the_input_is_enlarged_bilinear_to_the_target(self):
        seen, _ = self._run(factor=2.0)
        prepared = Image.open(seen["init_image"])
        self.assertEqual(prepared.size, (seen["width"], seen["height"]))
        self.assertGreater(prepared.width, 512)

    def test_enough_steps_to_actually_rebuild_detail(self):
        # 4 pas × 0,8 de débruitage = 3 pas utiles : trop court.
        seen, _ = self._run()
        self.assertGreaterEqual(seen["steps"], 6)


class VramFallbackTests(unittest.TestCase):
    def test_it_retries_smaller_instead_of_failing(self):
        tries = []

        def fake_generate(**call):
            tries.append((call["width"], call["height"]))
            if len(tries) < 2:
                raise sdcpp.VramError("plus de VRAM")
            out = Path(tempfile.mkdtemp()) / "out.png"
            Image.new("RGB", (call["width"], call["height"]), "navy").save(out)
            return [out]

        with patch.object(highres.gen_engine, "generate",
                          side_effect=fake_generate):
            highres.high_resolution(Image.new("RGB", (1024, 1024)),
                                    model_id="flux2-klein-9b", factor=3.0,
                                    match_colors=False)
        self.assertEqual(len(tries), 2)
        self.assertLess(tries[1][0] * tries[1][1], tries[0][0] * tries[0][1])

    def test_it_gives_up_instead_of_looping(self):
        with patch.object(highres.gen_engine, "generate",
                          side_effect=sdcpp.VramError("non")):
            with self.assertRaises(sdcpp.VramError):
                highres.high_resolution(Image.new("RGB", (1024, 1024)),
                                        model_id="flux2-klein-9b", factor=3.0)


class ModelChoiceTests(unittest.TestCase):
    def test_only_models_that_accept_a_reference_are_offered(self):
        from atelier import registry

        def model(mid, edit, ready):
            m = registry.BaseModel(
                id=mid, name=mid, family="flux2", tags=[], description="",
                components=[], defaults=({"edit": edit} if edit else {}),
                vram_min_gb=12, presets=[])
            return m, ready

        made = [model("klein", "full", True), model("krea", "optional", True),
                model("autre", None, True), model("pas-la", "full", False)]
        with patch.object(highres.registry, "load_base_models",
                          return_value=[m for m, _ in made]), \
             patch.object(highres.registry, "model_is_ready",
                          side_effect=lambda m: dict(
                              (x.id, r) for x, r in made)[m.id]):
            offered = highres.edit_models({})
        self.assertEqual([mid for _, mid in offered], ["klein"])


if __name__ == "__main__":
    unittest.main()
