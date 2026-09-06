"""Documentation des samplers/schedulers : complétude et cohérence.

Le risque réel n'est pas que la prose soit médiocre — c'est qu'une clé du menu
ne soit pas celle qu'attend `sd-cli`. Une faute de frappe passe la construction
de l'interface sans bruit et n'échoue qu'à la génération, avec un message du
moteur. Les listes de référence ci-dessous sont relevées dans la table de noms
du moteur (`sample_method_to_str` de src/stable-diffusion.cpp) et dans l'aide de
`--scheduler`.
"""
import re
import unittest
from pathlib import Path

from atelier import sampling

# Relevé dans sample_method_to_str[] (src/stable-diffusion.cpp).
ENGINE_SAMPLERS = {
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep",
    "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge",
    "dpm++2m_sde", "dpm++2m_sde_bt", "lms",
}
# Relevé dans l'aide de --scheduler (« auto » est notre valeur à nous : elle
# signifie « ne pas passer l'option », donc elle n'existe pas côté moteur).
ENGINE_SCHEDULERS = {
    "discrete", "karras", "exponential", "ays", "gits", "smoothstep",
    "sgm_uniform", "simple", "kl_optimal", "lcm", "bong_tangent", "ltx2",
    "logit_normal", "flux2", "flux", "beta",
}

# Les deux familles documentées — et, ce qui n'est pas un hasard, les deux
# sont distillées à CFG 1.0. Plusieurs verdicts ci-dessous découlent de CETTE
# propriété et non du modèle : si un modèle NON distillé revenait un jour au
# catalogue, ce sont ces tests-là qu'il faudrait rouvrir, pas les autres.
FAMILIES = ("flux2", "krea2")
DISTILLED = FAMILIES


class KeysMatchTheEngineTests(unittest.TestCase):
    def test_no_sampler_the_engine_would_refuse(self):
        unknown = set(sampling.SAMPLERS) - ENGINE_SAMPLERS
        self.assertEqual(unknown, set(), f"clés inconnues du moteur : {unknown}")

    def test_no_scheduler_the_engine_would_refuse(self):
        unknown = set(sampling.SCHEDULES) - ENGINE_SCHEDULERS - {"auto"}
        self.assertEqual(unknown, set(), f"clés inconnues du moteur : {unknown}")

    def test_every_engine_sampler_is_offered(self):
        """Un sampler que le moteur sait faire mais qu'on n'expose pas est une
        fonctionnalité perdue en silence."""
        missing = ENGINE_SAMPLERS - set(sampling.SAMPLERS)
        self.assertEqual(missing, set(), f"samplers non exposés : {missing}")


class DocumentationCompletenessTests(unittest.TestCase):
    def test_every_entry_is_fully_documented(self):
        for table, kind in ((sampling.SAMPLERS, "sampler"),
                            (sampling.SCHEDULES, "schedule")):
            for key, entry in table.items():
                name, summary, pro, con, levels = entry
                self.assertTrue(name, key)
                for field, txt in (("résumé", summary), ("avantage", pro),
                                   ("inconvénient", con)):
                    self.assertGreater(len(txt), 20,
                                       f"{kind} « {key} » : {field} trop court")
                for fam in FAMILIES:
                    self.assertIn(fam, levels, f"{kind} « {key} » : {fam} absent")

    def test_levels_are_valid(self):
        valid = {sampling.BEST, sampling.OK, sampling.MEH, sampling.BAD}
        for table in (sampling.SAMPLERS, sampling.SCHEDULES):
            for key, entry in table.items():
                for fam, lv in entry[4].items():
                    self.assertIn(lv, valid, f"{key}/{fam}")

    def test_each_family_has_exactly_one_recommended_sampler(self):
        """Deux « ⭐ » dans un menu, c'est ne pas répondre à la question."""
        for fam in FAMILIES:
            best = [k for k in sampling.SAMPLERS
                    if sampling.level("sampler", k, fam) == sampling.BEST]
            self.assertEqual(best, ["euler"], f"{fam} : {best}")


class VerdictsFollowTheModelTests(unittest.TestCase):
    """Les verdicts doivent découler des propriétés vérifiées du modèle."""

    _CFGPP = ("euler_cfg_pp", "euler_a_cfg_pp")

    def test_cfgpp_is_discouraged_on_the_distilled_models(self):
        # Ils tournent à CFG 1.0 : il n'y a aucun guidage à corriger, donc
        # rien à attendre de cette famille. Ce verdict tient à la DISTILLATION
        # et non aux modèles : il ne se recopie pas sur un modèle à CFG réel.
        for fam in DISTILLED:
            for key in self._CFGPP:
                self.assertEqual(sampling.level("sampler", key, fam),
                                 sampling.BAD, f"{key}/{fam}")

    def test_lcm_and_tcd_are_discouraged_everywhere(self):
        # Réservés aux modèles distillés PAR ces méthodes ; aucun des deux.
        for fam in FAMILIES:
            for key in ("lcm", "tcd"):
                self.assertEqual(sampling.level("sampler", key, fam),
                                 sampling.BAD, f"{key}/{fam}")
        for fam in FAMILIES:
            self.assertEqual(sampling.level("schedule", "lcm", fam),
                             sampling.BAD)

    def test_auto_is_recommended_for_both(self):
        for fam in FAMILIES:
            self.assertEqual(sampling.level("schedule", "auto", fam),
                             sampling.BEST)

    def test_flux2_scheduler_is_best_on_flux2_only(self):
        self.assertEqual(sampling.level("schedule", "flux2", "flux2"),
                         sampling.BEST)
        self.assertNotEqual(sampling.level("schedule", "flux2", "krea2"),
                            sampling.BEST)

    @staticmethod
    def _usable(fam):
        return sum(1 for k in sampling.SAMPLERS
                   if sampling.level("sampler", k, fam)
                   in (sampling.BEST, sampling.OK))

    def test_more_steps_open_more_samplers(self):
        """8 pas laissent plus de marge que 4 : le classement doit suivre ce
        budget, sinon les deux colonnes ne disent pas ce qu'elles prétendent."""
        self.assertGreater(self._usable("krea2"), self._usable("flux2"))


class RenderingTests(unittest.TestCase):
    def test_choices_are_annotated_and_keep_stable_values(self):
        for kind, table in (("sampler", sampling.SAMPLERS),
                            ("schedule", sampling.SCHEDULES)):
            for fam in FAMILIES:
                ch = sampling.choices(kind, fam)
                self.assertEqual([v for _l, v in ch], list(table))
                self.assertTrue(any("⭐" in lbl for lbl, _v in ch),
                                f"{kind}/{fam} : aucun recommandé marqué")

    def test_describe_covers_every_key(self):
        for kind, table in (("sampler", sampling.SAMPLERS),
                            ("schedule", sampling.SCHEDULES)):
            for key in table:
                for fam in FAMILIES:
                    txt = sampling.describe(kind, key, fam)
                    self.assertIn("✅", txt)
                    self.assertIn("❌", txt)
                    self.assertGreater(len(txt), 80, f"{kind}/{key}")

    def test_describe_is_empty_for_an_unknown_key(self):
        self.assertEqual(sampling.describe("sampler", "n'existe pas", "krea2"),
                         "")

    def test_rationale_is_model_specific(self):
        flux = sampling.rationale("flux2")
        krea = sampling.rationale("krea2")
        self.assertIn("Flux.2 Klein", flux)
        self.assertIn("4 steps", flux)
        self.assertIn("Krea 2 Turbo", krea)
        self.assertIn("8 steps", krea)
        self.assertNotEqual(flux, krea)

    def test_the_rationale_says_the_negative_prompt_is_ignored(self):
        """Les deux modèles sont distillés à CFG 1.0. Le dépliant doit le dire,
        parce que c'est la question qu'on pose en voyant le champ grisé."""
        for fam in FAMILIES:
            self.assertIn("negative prompt is\nignored", sampling.rationale(fam))


class ReadmeStaysInSyncTests(unittest.TestCase):
    """Le README recopie les verdicts dans deux tableaux. Une table recopiée
    à la main dérive au premier changement d'avis ; autant le vérifier."""

    _MARK_TO_LEVEL = {"⭐": sampling.BEST, "✓": sampling.OK,
                      "△": sampling.MEH, "⚠️": sampling.BAD}

    # Une colonne par famille documentée, dans l'ordre des tableaux du README.
    _COLUMNS = FAMILIES

    def _readme_tables(self) -> dict[str, dict[str, tuple[str, ...]]]:
        readme = (Path(__file__).resolve().parent.parent / "README.md")
        kind, out = None, {"sampler": {}, "schedule": {}}
        width = 1 + len(self._COLUMNS)
        for line in readme.read_text(encoding="utf-8").splitlines():
            if line.startswith("| Sampler |"):
                kind = "sampler"
                continue
            if line.startswith("| Scheduler |"):
                kind = "schedule"
                continue
            if not line.startswith("|"):
                kind = None
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if kind is None or set(cells[0]) <= {"-", ":"}:
                continue
            # Une ligne large ou étroite est une colonne oubliée, pas une ligne
            # à ignorer : l'ancienne version sautait en silence.
            self.assertEqual(len(cells), width, f"tableau {kind} : {line}")
            keys = re.findall(r"`([^`]+)`", cells[0])
            self.assertTrue(keys, line)
            for key in keys:
                out[kind][key] = tuple(cells[1:])
        return out

    def test_every_option_appears_exactly_once(self):
        tables = self._readme_tables()
        for kind, table in (("sampler", sampling.SAMPLERS),
                            ("schedule", sampling.SCHEDULES)):
            self.assertEqual(set(tables[kind]), set(table),
                             f"tableau {kind} du README désynchronisé")

    def test_readme_marks_match_the_code(self):
        tables = self._readme_tables()
        wrong = []
        for kind in ("sampler", "schedule"):
            for key, marks in tables[kind].items():
                for mark, fam in zip(marks, self._COLUMNS):
                    got = self._MARK_TO_LEVEL.get(mark)
                    self.assertIsNotNone(got, f"marqueur inconnu : {mark!r}")
                    if got != sampling.level(kind, key, fam):
                        wrong.append(f"{kind}/{key}/{fam} : README dit {mark}, "
                                     f"le code dit "
                                     f"{sampling.level(kind, key, fam)}")
        self.assertEqual(wrong, [], "\n".join(wrong))


if __name__ == "__main__":
    unittest.main()
