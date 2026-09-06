"""L'interface reste-t-elle praticable pour quelqu'un qui n'est pas dev ?

Ces tests ne jugent pas le goût. Ils mesurent trois choses qui se dégradent
TOUJOURS de la même façon quand on ajoute des fonctionnalités une par une :

1. **la profondeur d'imbrication** — un réglage enterré sous trois dépliages
   n'existe pas ; on ne le trouve qu'en le cherchant, donc jamais ;
2. **le nombre de blocs à ouvrir soi-même** sur une machine où tout est déjà
   installé — c'est l'état permanent de l'utilisateur, pas l'état du premier
   jour ;
3. **les blocs qui ne servent qu'une fois** (les installateurs) et qui
   devraient disparaître une fois l'outil en place.

Deux natures de seuil, volontairement : les COMPTAGES sont laissés au-dessus
de l'état actuel (le test n'est pas là pour figer un chiffre, mais pour qu'une
dérive franche fasse du bruit), tandis que l'imbrication est à ZÉRO — un repli
dans un repli n'est jamais acceptable, donc il n'y a pas de marge à donner.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from atelier.engine import tools  # noqa: E402


def _build(all_installed: bool):
    """Construit l'application en simulant une machine donnée.

    On bascule les sondes `*_is_installed` : c'est exactement ce dont dépend la
    visibilité des blocs d'installation, et c'est la seule différence entre le
    premier lancement et la vie courante.
    """
    saved = {}
    if all_installed:
        for name in dir(tools):
            if name.endswith("_is_installed"):
                saved[name] = getattr(tools, name)
                setattr(tools, name, lambda *a, **k: True)
    try:
        import app
        return app.build_app()
    finally:
        for name, fn in saved.items():
            setattr(tools, name, fn)


def _accordions(demo) -> list[gr.Accordion]:
    return [b for b in demo.blocks.values() if isinstance(b, gr.Accordion)]


class AccordionBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fresh = _build(all_installed=False)
        cls.settled = _build(all_installed=True)

    def _visible(self, demo):
        return [a for a in _accordions(demo) if getattr(a, "visible", True)]

    def test_installers_disappear_once_the_tool_is_there(self):
        """Le cœur du problème : sept blocs « Installer … » qui restaient à vie.

        On compare la MÊME application dans deux états ; si les installateurs
        ne se masquaient plus, les deux chiffres seraient égaux.
        """
        fresh = len(self._visible(self.fresh))
        settled = len(self._visible(self.settled))
        self.assertLess(settled, fresh,
                        "aucun bloc d'installation ne disparaît une fois "
                        "les outils installés")
        self.assertGreaterEqual(fresh - settled, 5)

    def test_the_settled_machine_has_few_things_to_unfold(self):
        """C'est l'état dans lequel l'utilisateur passe sa vie."""
        visible = self._visible(self.settled)
        self.assertLessEqual(
            len(visible), 36,
            "trop de blocs à déplier :\n" +
            "\n".join(f"  - {a.label}" for a in visible))

    def test_no_accordion_lives_inside_another(self):
        """AUCUN accordéon imbriqué : le seuil est zéro, pas « raisonnable ».

        Un repli dans un repli demande deux clics pour voir une option, et le
        second n'est pas annoncé — c'est précisément la cascade qu'on vient de
        retirer. Là où plusieurs rubriques doivent cohabiter, ce sont des
        ONGLETS : ils se voient tous sans rien ouvrir. Le seuil à zéro est ce
        qui donne sa valeur au test ; l'assouplir le rendrait décoratif.

        La profondeur se mesure sur l'arbre réel de Gradio et non sur le code :
        c'est l'imbrication rendue qui compte pour celui qui clique.
        """
        nested = []
        for acc in self._visible(self.settled):
            node = acc.parent
            while node is not None:
                if isinstance(node, gr.Accordion):
                    nested.append(f"« {acc.label} » est replié dans "
                                  f"« {node.label} »")
                    break
                node = node.parent
        self.assertEqual(nested, [], "\n".join(nested))


class GenerationTabTests(unittest.TestCase):
    """L'onglet le plus utilisé : rien d'expérimental ne doit s'y inviter."""

    def test_the_int8_variant_stays_hidden_until_it_is_installed(self):
        """Proposer un modèle non téléchargé, c'est un piège : on clique, la
        génération échoue. Le sélecteur n'existe que si le fichier est là."""
        demo = _build(all_installed=False)
        radios = [b for b in demo.blocks.values()
                  if isinstance(b, gr.Radio)
                  and (b.label or "").startswith("Format du modèle")]
        self.assertEqual(radios, [])


if __name__ == "__main__":
    unittest.main()


class SettingsTabTests(unittest.TestCase):
    """Les réglages : une décision, pas un formulaire.

    L'onglet a déjà dérivé deux fois — six accordéons empilés, puis quatre
    onglets. Les deux fois, la plainte était la même : on ne sait pas quoi
    cocher. Ces tests fixent ce qui empêche la dérive de recommencer.
    """

    @classmethod
    def setUpClass(cls):
        cls.demo = _build(all_installed=True)
        cls.settings = [b for b in cls.demo.blocks.values()
                        if isinstance(b, gr.Tab)
                        and (b.label or "").endswith("Settings")]

    def _inside(self, kinds):
        """Composants rendus à l'intérieur de l'onglet Réglages."""
        self.assertEqual(len(self.settings), 1, "onglet Réglages introuvable")
        root = self.settings[0]
        out = []
        for b in self.demo.blocks.values():
            node = getattr(b, "parent", None)
            while node is not None:
                if node is root:
                    if isinstance(b, kinds):
                        out.append(b)
                    break
                node = getattr(node, "parent", None)
        return out

    def test_no_tabs_inside_the_settings(self):
        """Des onglets dans un onglet dans un onglet : on ne sait plus où on est.

        « Système > Réglages > Accélération » faisait trois niveaux pour
        atteindre une case à cocher, et rien ne disait laquelle regarder en
        premier. Un écran, une lecture de haut en bas.
        """
        nested = [b.label for b in self._inside(gr.Tab)]
        self.assertEqual(nested, [], f"onglets imbriqués : {nested}")

    def test_only_optional_things_are_folded(self):
        """Ce qui est replié doit être facultatif, et le dire dans son titre."""
        labels = [b.label or "" for b in self._inside(gr.Accordion)]
        self.assertLessEqual(len(labels), 3, labels)
        for lbl in labels:
            self.assertTrue(
                any(w in lbl for w in ("facultatif", "optional", "Détail",
                                       "details", "Theme", "accounts")),
                f"repli sans promesse d'être secondaire : « {lbl} »")

    def test_exactly_one_decision_is_asked_up_front(self):
        """Hors repli, il ne reste QUE le curseur qualité/mémoire.

        Le sélecteur de carte et la répartition n'apparaissent qu'avec deux
        cartes ; sur une machine mono-GPU (le cas de ce test) ils n'existent
        pas, et il ne doit plus rien rester d'autre à décider.
        """
        folded = set()
        for acc in self._inside(gr.Accordion):
            for b in self.demo.blocks.values():
                node = getattr(b, "parent", None)
                while node is not None:
                    if node is acc:
                        folded.add(id(b))
                        break
                    node = getattr(node, "parent", None)
        inputs = [b for b in self._inside((gr.Radio, gr.Checkbox, gr.Dropdown))
                  if id(b) not in folded and getattr(b, "visible", True)]
        self.assertEqual(
            [type(b).__name__ for b in inputs], ["Radio"],
            "réglages visibles d'emblée : " +
            ", ".join(f"{type(b).__name__}({b.label})" for b in inputs))

    def test_there_is_no_save_button_to_forget(self):
        """Tout s'applique à la volée : un bouton « Enregistrer » rouvrirait la
        question « est-ce que ça a été pris en compte ? » — d'autant que la
        langue et le thème, eux, s'enregistraient déjà tout seuls."""
        buttons = [(b.value or "") for b in self._inside(gr.Button)]
        self.assertEqual(
            [v for v in buttons if "Enregistrer" in v or "Save" in v], [])
