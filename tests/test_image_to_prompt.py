"""« Image → prompt » : ce qui distingue un prompt d'une légende.

Le modèle n'est pas testable ici (7,5 Go de poids, un GPU, et une sortie non
déterministe). Ce qui EST testable, et qui décide de la qualité du résultat,
c'est tout ce qu'il y a autour : les consignes envoyées au modèle, le nettoyage
de sa réponse, et le câblage qui ramène le texte dans le champ Prompt.

Ces trois choses ont chacune un mode de panne silencieux :
- une consigne qui n'interdit pas les formules de légende → « This image shows
  a cat », collé tel quel, donne une image plate ;
- un nettoyage qui laisse passer l'amorce → même résultat, malgré la consigne ;
- un câblage qui n'écrit nulle part → le bouton semble marcher et ne fait rien
  (c'est arrivé : la première version passait par un State consommé au
  changement d'onglet, et une sélection programmatique ne déclenche pas
  `Tabs.select`).
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

ROOT = Path(__file__).resolve().parent.parent


def _runner():
    """Charge run_describe.py SANS importer torch (absent des tests)."""
    spec = importlib.util.spec_from_file_location(
        "run_describe", ROOT / "scripts" / "tools" / "run_describe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # les imports lourds sont dans main()
    return mod


class CleaningTests(unittest.TestCase):
    """La consigne réduit les amorces de légende ; elle ne les supprime pas."""

    def setUp(self):
        self.run = _runner()

    def test_caption_lead_ins_are_stripped(self):
        cases = {
            "This image shows a red fox in the snow":
                "A red fox in the snow",
            "The picture depicts an old man reading":
                "An old man reading",
            "Here is a prompt: cinematic portrait, soft light":
                "Cinematic portrait, soft light",
            "Sure, a moody landscape at dusk": "A moody landscape at dusk",
        }
        for raw, expected in cases.items():
            self.assertEqual(self.run._clean(raw), expected, raw)

    def test_stacked_lead_ins_are_all_removed(self):
        """Les modèles en empilent deux : « Sure, here is a prompt: … »."""
        got = self.run._clean("Sure, here is a prompt: golden hour photograph")
        self.assertEqual(got, "Golden hour photograph")

    def test_code_fences_and_quotes_go_away(self):
        self.assertEqual(
            self.run._clean('```\n"oil painting, thick impasto"\n```'),
            "Oil painting, thick impasto")

    def test_a_legitimate_prompt_is_left_alone(self):
        """Le nettoyage ne doit pas mordre sur un texte déjà correct."""
        good = ("Cinematic photograph of a lighthouse, 85mm lens, shallow "
                "depth of field, cold blue dusk light")
        self.assertEqual(self.run._clean(good), good)

    def test_empty_stays_empty(self):
        self.assertEqual(self.run._clean(""), "")
        self.assertEqual(self.run._clean("   "), "")


class ModePromptTests(unittest.TestCase):
    """Les trois modes doivent demander trois choses DIFFÉRENTES."""

    def setUp(self):
        self.run = _runner()

    def test_the_three_modes_exist_and_differ(self):
        self.assertEqual(set(self.run.MODES), {"full", "style", "plain"})
        self.assertEqual(len(set(self.run.MODES.values())), 3)

    def test_english_is_demanded_everywhere(self):
        """L'interface est en français ; les modèles, non. Sans cette consigne
        le VLM répond dans la langue de la question."""
        for name, text in self.run.MODES.items():
            self.assertIn("ENGLISH", text, name)

    def test_caption_formulas_are_forbidden_everywhere(self):
        for name, text in self.run.MODES.items():
            self.assertIn("NEVER start with caption formulas", text, name)

    def test_style_mode_forbids_naming_the_subject(self):
        """C'est TOUT l'intérêt du mode : un style réutilisable ailleurs. S'il
        nomme le sujet, il n'est pas transposable et le mode ne sert à rien."""
        style = self.run.MODES["style"]
        self.assertIn("Say NOTHING about what the picture is of", style)
        self.assertIn("ONLY THE STYLE", style)

    def test_full_mode_asks_for_what_actually_steers_diffusion(self):
        full = self.run.MODES["full"]
        for axis in ("LIGHT", "COMPOSITION", "COLOURS", "MEDIUM"):
            self.assertIn(axis, full, axis)

    def test_plain_mode_does_not_ask_for_prompt_jargon(self):
        plain = self.run.MODES["plain"]
        self.assertIn("No prompt vocabulary", plain)

    def test_medium_coherence_is_required(self):
        """Mélanger « oil painting » et « 85mm lens » brouille le rendu : c'est
        la faute la plus commune d'un prompt écrit par une machine, et elle a
        été observée en vrai. La consigne doit nommer les termes interdits, pas
        se contenter d'un principe."""
        for name, text in self.run.MODES.items():
            self.assertIn("Decide the MEDIUM once", text, name)
            self.assertIn("brushstrokes", text, name)

    def test_the_full_mode_asks_for_the_medium_FIRST(self):
        """La dérive arrive en FIN de génération. Faire nommer le médium en
        premier le fixe pendant que le modèle regarde encore l'image."""
        full = self.run.MODES["full"]
        self.assertIn("1. the MEDIUM, first", full)
        self.assertLess(full.index("the MEDIUM, first"),
                        full.index("the main subject"))

    def test_colour_honesty_is_demanded(self):
        """« dark red shoes » sur des escarpins noirs : la couleur inventée est
        la seconde erreur observée."""
        for name, text in self.run.MODES.items():
            self.assertIn("COLOURS above all", text, name)


class WiringTests(unittest.TestCase):
    """Le bouton « → Krea 2 » doit VRAIMENT écrire dans le champ Prompt."""

    def test_generation_tabs_return_their_prompt_box(self):
        """C'est ce que l'application collecte pour câbler l'envoi. Si un jour
        `build_generative_tab` cesse de le renvoyer, le bouton d'envoi
        disparaîtrait en silence au lieu d'échouer."""
        import gradio as gr

        import app
        demo = app.build_app()
        boxes = [b for b in demo.blocks.values()
                 if isinstance(b, gr.Textbox) and b.label == "Prompt"]
        self.assertGreaterEqual(len(boxes), 2)

    def test_the_send_buttons_exist_and_are_visible(self):
        import gradio as gr

        import app
        demo = app.build_app()
        labels = [(b.value or "") for b in demo.blocks.values()
                  if isinstance(b, gr.Button) and getattr(b, "visible", True)]
        self.assertTrue(any("Krea 2 Turbo" in v and v.startswith("→")
                            for v in labels), labels[:5])
        self.assertTrue(any("Flux.2 Klein" in v and v.startswith("→")
                            for v in labels))

    def test_the_tool_is_reachable_from_the_toolkit(self):
        import gradio as gr

        import app
        demo = app.build_app()
        ids = [getattr(b, "id", None) for b in demo.blocks.values()
               if isinstance(b, gr.Tab)]
        self.assertIn("describe", ids)
        # Les onglets de génération ont besoin d'un id STABLE : l'envoi les
        # vise par identifiant, pas par libellé (qui change avec la langue).
        self.assertIn("krea2-turbo", ids)
        self.assertIn("flux2-klein-9b", ids)


class EngineLayerTests(unittest.TestCase):
    def test_modes_agree_between_the_app_and_the_runner(self):
        """Deux listes de modes qui divergent = un mode qui échoue à l'appel,
        avec un message d'argparse que personne ne comprendra."""
        from atelier.engine import tools
        self.assertEqual(set(tools.DESCRIBE_MODES), set(_runner().MODES))

    def test_it_refuses_politely_when_not_installed(self):
        from atelier.engine import tools
        if tools.describe_is_installed():
            self.skipTest("modèle installé sur cette machine")
        with self.assertRaises(tools.ToolError) as ctx:
            tools.image_to_prompt(ROOT / "README.md")
        self.assertIn("not installed", str(ctx.exception))


# Sortie RÉELLE remontée par l'utilisateur : bon début, puis le modèle s'enferme
# dans une boucle et répète « high heels, fashion, modern, wet, rain » jusqu'à
# épuiser son budget de jetons. C'est la panne à couvrir, avec ses vraies
# données — pas un exemple reconstitué qui aurait pu être plus commode.
LOOPED_OUTPUT = (
    "Shiny black high heels, woman's legs, mid-length, slim figure, wearing form-fitting dress, metallic finish, wet street, puddle, rainy night, city lights, bokeh effect, warm yellow-orange light, low angle, close-up, shallow depth of field, vivid colors, high contrast, glossy surface, urban scene, photograph, real texture, wet pavement, raindrops, shiny heels, splash, dark background, wet surface, reflection, high heels, fashion, evening, night, cityscape, street, sidewalk, wet, reflective, bokeh, night lights, vibrant, glossy, modern, sleek, high fashion, urban, rain, close-up, wet street, puddle, rain, splash, shiny, close-up, detail, wet, rain, dark, rainy, wet pavement, glossy, reflective, puddle, splash, shine, high heels, fashion, modern, sleek, wet, fashion, modern, wet, fashion, modern, high heels, wet, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet, rain, high heels, fashion, modern, wet")


class RepetitionLoopTests(unittest.TestCase):
    """Un modèle qui boucle est la panne NORMALE de ce format.

    Une liste de mots-clés séparés par des virgules ne dit jamais au modèle
    qu'il a fini : rien dans la grammaire ne signale la fin. Les pénalités
    passées au générateur réduisent le phénomène ; elles ne le suppriment pas,
    donc le nettoyage doit garantir le résultat plutôt que l'espérer.
    """

    def setUp(self):
        self.run = _runner()

    def test_the_real_looped_output_is_cleaned_up(self):
        out = self.run._clean(LOOPED_OUTPUT)
        before = LOOPED_OUTPUT.count(",") + 1
        after = out.count(",") + 1
        self.assertLess(after, before / 2,
                        f"{before} segments -> {after}, la boucle survit")

    def test_no_fragment_appears_twice(self):
        out = self.run._clean(LOOPED_OUTPUT)
        keys = [self.run._key(f) for f in out.split(",") if self.run._key(f)]
        self.assertEqual(len(keys), len(set(keys)),
                         "un fragment est encore présent deux fois")

    def test_the_useful_beginning_survives_intact(self):
        """Dédupliquer ne doit pas coûter la partie utile : c'est le DÉBUT qui
        porte le sujet, la lumière et l'objectif."""
        out = self.run._clean(LOOPED_OUTPUT)
        for kept in ("Shiny black high heels", "wearing form-fitting dress",
                     "warm yellow-orange light", "shallow depth of field",
                     "bokeh effect", "low angle", "photograph"):
            self.assertIn(kept, out, kept)

    def test_order_is_preserved(self):
        """L'ordre porte du sens : le sujet d'abord, les modificateurs après."""
        out = self.run._clean(LOOPED_OUTPUT)
        self.assertLess(out.index("high heels"), out.index("bokeh"))

    def test_variants_of_the_same_fragment_collapse(self):
        """« wet pavement » et « the wet pavement » sont le même segment."""
        got = self.run._dedupe("wet pavement, city lights, the wet pavement")
        self.assertEqual(got, "wet pavement, city lights")

    def test_a_sentence_is_never_deduplicated(self):
        """Le mode « décrire simplement » rend des PHRASES. Y couper des
        segments entre virgules casserait la grammaire, donc le nettoyage ne
        s'applique qu'aux listes."""
        prose = ("A woman walks through a puddle at night, her heels splashing "
                 "water, and the street lights glow behind her, warm and "
                 "diffuse, while the rain keeps falling.")
        self.assertEqual(self.run._clean(prose), prose)

    def test_a_truncated_tail_is_dropped_only_when_it_really_is(self):
        """On ne DEVINE pas la troncature : « shallow dep » et « shallow » sont
        indiscernables sans dictionnaire, et couper un fragment légitime est
        pire que laisser un moignon. Le runner sait si le modèle a été arrêté
        par la limite de jetons ; il le dit."""
        txt = "cinematic photograph, warm light, low angle, bokeh, shallow dep"
        self.assertEqual(self.run._clean(txt, truncated=True),
                         "Cinematic photograph, warm light, low angle, bokeh")
        # Sans le signal, on ne touche à rien.
        self.assertTrue(self.run._clean(txt).endswith("shallow dep"))

    def test_the_runner_detects_truncation_from_the_end_token(self):
        """Le signal vient de l'absence de jeton de fin, pas d'une heuristique
        sur le texte."""
        src = (ROOT / "scripts" / "tools"
               / "run_describe.py").read_text(encoding="utf-8")
        self.assertIn("eos_token_id", src)
        self.assertIn("truncated=cut", src)


class GenerationGuardTests(unittest.TestCase):
    """Les garde-fous côté génération, lus dans le code du runner.

    Ils ne sont pas testables sans GPU, mais leur ABSENCE est ce qui a produit
    la boucle : les vérifier statiquement vaut mieux que de les croire acquis.
    """

    def setUp(self):
        self.src = (ROOT / "scripts" / "tools"
                    / "run_describe.py").read_text(encoding="utf-8")

    def test_repetition_penalties_are_passed_to_generate(self):
        self.assertIn("repetition_penalty=", self.src)
        self.assertIn("no_repeat_ngram_size=", self.src)

    def test_the_token_budget_matches_the_requested_length(self):
        """320 jetons pour 110 mots laissaient 65 % de marge — et cette marge,
        le modèle la remplit de redites."""
        import re as _re
        m = _re.search(r'budget = \{([^}]*)\}', self.src)
        self.assertIsNotNone(m)
        budgets = {k: int(v) for k, v in
                   _re.findall(r'"(\w+)": (\d+)', m.group(1))}
        self.assertLessEqual(budgets["full"], 220, budgets)
        self.assertLess(budgets["style"], budgets["full"])


# Seconde sortie RÉELLE remontée : sur une PHOTO, le modèle termine par « oil
# painting style, brushstrokes visible, canvas texture evident ». Trois
# fragments qui contredisent tout ce qui précède — et qui suffisent à faire
# produire une peinture au modèle de diffusion.
MIXED_MEDIUM_OUTPUT = (
    'Wet woman legs wearing high heels, close-up on shiny pavement puddle, nighttime city street lights reflecting, soft focus background, dark red shoes, wet surface glistening, high contrast lighting, low angle shot, shallow depth of field, vibrant neon colors, oil painting style, brushstrokes visible, canvas texture evident.')


class MediumDriftTests(unittest.TestCase):
    """Un seul médium par prompt, garanti après coup.

    La consigne l'interdit déjà, en majuscules. Ça n'a pas suffi et ça ne
    suffira pas : la dérive arrive en FIN de génération, quand le modèle n'a
    plus rien de réel à dire. Même leçon que pour la boucle — on vérifie au
    lieu d'espérer.
    """

    def setUp(self):
        self.run = _runner()

    def test_the_real_painting_tail_is_removed_from_a_photo(self):
        out = self.run._clean(MIXED_MEDIUM_OUTPUT)
        for wrong in ("oil painting", "brushstrokes", "canvas texture"):
            self.assertNotIn(wrong, out.lower(), wrong)

    def test_everything_else_survives(self):
        """Retirer le médium fautif ne doit rien coûter d'autre."""
        out = self.run._clean(MIXED_MEDIUM_OUTPUT)
        for kept in ("Wet woman legs wearing high heels", "shiny pavement",
                     "nighttime city street lights", "low angle shot",
                     "shallow depth of field", "vibrant neon colors"):
            self.assertIn(kept, out, kept)

    def test_the_first_medium_named_is_the_one_kept(self):
        """Le premier gagne, et ce n'est pas arbitraire : le modèle décrit
        d'abord ce qu'il voit, la confabulation vient après."""
        painting = ("oil painting, thick impasto, warm palette, wooden frame, "
                    "shot on 85mm lens, shallow depth of field")
        out = self.run._one_medium(painting)
        self.assertIn("oil painting", out)
        self.assertNotIn("85mm lens", out)

    def test_a_single_medium_is_left_untouched(self):
        clean = ("cinematic photograph, 85mm lens, shallow depth of field, "
                 "warm rim light, wet asphalt, amber bokeh")
        self.assertEqual(self.run._one_medium(clean), clean)

    def test_a_prompt_with_no_medium_at_all_is_left_untouched(self):
        """Rien à arbitrer : on ne retire rien."""
        neutral = "wet asphalt, amber reflections, low angle, high contrast"
        self.assertEqual(self.run._one_medium(neutral), neutral)

    def test_ambiguous_words_are_not_used_as_markers(self):
        """« texture » seul ne désigne aucun médium ; « canvas texture » si.
        Un marqueur ambigu ferait supprimer des fragments légitimes."""
        self.assertIsNone(self.run._medium_of("rich surface texture"))
        self.assertIsNone(self.run._medium_of("vivid colors"))
        self.assertEqual(self.run._medium_of("canvas texture evident"),
                         "painting")


class SamplingTests(unittest.TestCase):
    """Décrire n'est pas créer.

    Le tirage aléatoire demande au modèle de choisir parfois un jeton MOINS
    probable — donc, sur une description, d'inventer. « dark red shoes » sur des
    escarpins noirs en est le résultat direct. Sur une seule proposition il ne
    doit pas y avoir de tirage du tout.
    """

    def setUp(self):
        self.src = (ROOT / "scripts" / "tools"
                    / "run_describe.py").read_text(encoding="utf-8")

    def test_a_single_proposal_is_generated_greedily(self):
        self.assertIn('{"do_sample": False} if n == 1', self.src)

    def test_several_proposals_stay_cold(self):
        """Le tirage ne revient que pour VARIER — et à température basse : on
        veut des formulations différentes, pas des faits différents."""
        import re as _re
        m = _re.search(r'"temperature": ([\d.]+)', self.src)
        self.assertIsNotNone(m, "plus de température dans le code ?")
        self.assertLessEqual(float(m.group(1)), 0.4)


if __name__ == "__main__":
    unittest.main()
