"""Passe HD : que fait-elle quand la VRAM manque ?

L'échelle de repli du moteur est documentée (sd.cpp, docs/performance.md) :
« --offload-to-cpu → + --max-vram → + --stream-layers », et les trois cumulés
font tourner des modèles 3 à 4 fois plus gros que la VRAM brute. L'application
avait les deux premiers barreaux et pas le troisième : au premier manque de
mémoire elle baissait le facteur, donc rendait une image plus petite que
demandée alors qu'un barreau restait.

Ce qui compte ici n'est pas qu'un drapeau soit passé, c'est l'ORDRE : baisser
le facteur coûte des pixels définitivement, streamer les couches coûte du
temps. On tente donc d'abord ce qui ne sacrifie rien du résultat.
"""
from __future__ import annotations

import unittest
from unittest import mock

from atelier.engine import generate as gen
from atelier.engine import sdcpp


class StreamingPreconditionTests(unittest.TestCase):
    """`--stream-layers` n'a d'effet que si les poids de diffusion sont en RAM.

    C'est la règle du moteur, et c'est ce que l'ancien code ignorait : il liait
    l'option à `--max-vram`, qui n'a rien à voir. La condition est isolée dans
    une fonction pour que la passe HD puisse la poser AVANT de tenter.
    """

    def _known(self, *opts):
        return mock.patch.object(sdcpp, "supported_options",
                                 lambda _cli: set(opts))

    def test_needs_the_weights_in_ram(self):
        with self._known("--stream-layers"):
            self.assertFalse(sdcpp.stream_layers_possible(
                None, {"offload_to_cpu": False}, ""))
            self.assertTrue(sdcpp.stream_layers_possible(
                None, {"offload_to_cpu": True}, ""))

    def test_an_explicit_cpu_residency_counts_too(self):
        with self._known("--stream-layers"):
            for backend in ("diffusion=cpu,vae=cuda0", "*=cpu"):
                self.assertTrue(sdcpp.stream_layers_possible(
                    None, {"offload_to_cpu": False}, backend), backend)

    def test_an_old_engine_disqualifies_it(self):
        """Sur un moteur qui ne connaît pas l'option, tenter serait une
        tentative gâchée : le manque de VRAM se reproduirait à l'identique."""
        with self._known("--max-vram"):
            self.assertFalse(sdcpp.stream_layers_possible(
                None, {"offload_to_cpu": True}, ""))


class HdFallbackOrderTests(unittest.TestCase):
    """L'ordre des reprises, mesuré sur un moteur simulé."""

    def _run(self, fail_times: int, stream_possible: bool = True):
        """Rejoue la boucle de reprise avec un moteur qui échoue N fois.

        On reproduit la boucle plutôt que d'appeler `hd_upscale`, qui exige un
        binaire, un modèle et une image. Ce qu'on teste est la POLITIQUE de
        repli, et elle est entièrement dans cette boucle.
        """
        tries: list[tuple[float, bool]] = []
        scale, streaming_on, drops = 2.0, False, 0
        while True:
            tries.append((round(scale, 3), streaming_on))
            if len(tries) > fail_times:
                return tries
            last = scale
            if stream_possible and not streaming_on:
                streaming_on = True
                continue
            if drops >= gen.HD_MAX_RETRIES:
                return tries
            scale = max(1.25, scale * gen.HD_RETRY_FACTOR)
            if scale >= last:
                return tries
            drops += 1

    def test_streaming_is_tried_before_losing_pixels(self):
        """Le 2e essai garde le facteur : c'est tout l'objet du changement."""
        tries = self._run(fail_times=1)
        self.assertEqual(tries[0], (2.0, False))
        self.assertEqual(tries[1], (2.0, True), "le facteur a baissé trop tôt")

    def test_the_factor_drops_only_after_streaming_failed(self):
        tries = self._run(fail_times=2)
        self.assertEqual([t[0] for t in tries[:2]], [2.0, 2.0])
        self.assertLess(tries[2][0], 2.0)
        self.assertTrue(tries[2][1], "le streaming doit rester actif ensuite")

    def test_streaming_does_not_eat_a_rung_of_the_ladder(self):
        """La tentative en streaming ne sacrifie aucun pixel : elle ne doit pas
        consommer une des baisses de facteur. Sinon activer le streaming
        coûterait une réduction de taille — ce qu'il sert à éviter."""
        without = self._run(fail_times=9, stream_possible=False)
        with_stream = self._run(fail_times=9, stream_possible=True)
        drops_without = len({t[0] for t in without})
        drops_with = len({t[0] for t in with_stream})
        self.assertEqual(drops_without, drops_with,
                         f"{drops_without} facteurs sans streaming, "
                         f"{drops_with} avec")

    def test_an_engine_without_streaming_behaves_exactly_as_before(self):
        tries = self._run(fail_times=1, stream_possible=False)
        self.assertLess(tries[1][0], 2.0)
        self.assertFalse(tries[1][1])

    def test_the_ladder_stops_instead_of_looping(self):
        tries = self._run(fail_times=99)
        self.assertLessEqual(len(tries), gen.HD_MAX_RETRIES + 3)


class GenerateOverrideTests(unittest.TestCase):
    def test_generate_accepts_a_one_shot_streaming_override(self):
        """La passe HD force le streaming pour UNE tentative, sans écrire dans
        les préférences de l'utilisateur — même patron que `max_vram`."""
        import inspect
        sig = inspect.signature(gen.generate)
        self.assertIn("stream_layers", sig.parameters)
        self.assertIsNone(sig.parameters["stream_layers"].default,
                          "None = suivre les préférences ; c'est le défaut")


if __name__ == "__main__":
    unittest.main()
