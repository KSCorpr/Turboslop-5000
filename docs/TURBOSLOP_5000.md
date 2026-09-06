# Turboslop 5000 — modifications et validation

Copie de `KSCorpr/Turbo-Slop-Generator-3000`, commit de départ
`2e83c762e439853d93b9932969c249163ee64c4d`. Travail du 6 septembre 2026.
Version de l'application : `5.0.0`. Aucune modification envoyée au dépôt 3000.

La philosophie est conservée : application locale et portable, interface Gradio,
génération native via stable-diffusion.cpp, modèles GGUF à la demande, outils
Python lourds installés seulement lorsqu'ils sont utilisés. Aucun nouveau moteur,
service cloud ni dépendance de production ajouté. Les six onglets racine et
les outils existants sont conservés. Les crédits du projet initial restent présents.

## Changements effectifs

| Zone | Problème traité | Comportement de 5000 |
|---|---|---|
| Moteur natif | L'option auto-fit a changé de syntaxe entre la release et le code amont | Détection de l'aide du binaire : ancien commutateur ou `on/off` explicite |
| Placement mémoire | CLI et serveur dupliquaient une logique susceptible de diverger | Construction commune des paramètres mémoire |
| Reprise après OOM | La résidence explicite supprimait le choix du calcul CPU de l'encodeur | Conservation du calcul CPU et déplacement de ses poids en RAM lors de la reprise |
| Moteurs récents | Le streaming explicite n'existe plus dans le dernier code inspecté | Pas d'envoi d'un drapeau supprimé ; segmentation/préchargement automatiques laissés au moteur |
| Serveur résident | Le CFG était envoyé comme nombre, alors que l'API attend un objet | `sample_params.guidance.txt_cfg` transmet réellement le CFG demandé |
| Édition par références | Retour systématique à sd-cli, avec rechargement des poids | Réutilisation du serveur si l'API annonce `ref_images`, avec projecteur vision si nécessaire |
| Annulation | L'API ne sait pas annuler un travail déjà en calcul | Arrêt du processus, attente de sa terminaison, aucune relance CLI après annulation |
| Robustesse du serveur | Attente sans fin et remplacement de modèle concurrent possibles | Délai maximal de deux heures, exclusion des requêtes concurrentes, détection des erreurs mémoire |
| Modèle résident | Remplacer les poids au même chemin ne changeait pas sa clé | Taille et date du fichier participent à l'identité de la session |
| Réseau local | Les paramètres proxy pouvaient affecter le client local | Les échanges avec 127.0.0.1 contournent les proxies système |
| Catalogue | Le YAML était relu et analysé à chaque appel | Cache invalidé par modification du fichier ; copies indépendantes pour les appelants |
| Téléchargements | Le Hub était interrogé même pour des poids déjà installés | Vérification locale d'abord ; une quantification inférieure ne bloque pas une mise à niveau |
| Téléchargements | Plusieurs composants relistaient le même dépôt | Une liste par dépôt pendant un téléchargement de modèle |
| Préférences et styles | Une interruption pouvait laisser un JSON partiellement écrit | Écriture temporaire sur le même disque, puis remplacement atomique |
| Benchmark matériel | Krea était mesuré en quatre pas ; le mode résident pouvait fausser les résultats | Nombre de pas natif, moteur résident arrêté, caches désactivés et paramètres consignés |
| Identité | Les mises à jour auraient réinstallé 3000 | Nom, version, archive portable et dépôt de mise à jour renommés pour 5000 |

Le moteur résident reste optionnel : il n'offre pas l'aperçu pas à pas. Les LoRA,
la passe HD, le cache entre pas et l'auto-fit continuent à utiliser la voie CLI.
La prise en charge des références a été vérifiée au niveau du protocole ; sa
qualité visuelle et sa performance nécessitent encore un essai avec de vrais poids.
Changer de prompt nécessite toujours son encodage : garder les poids en mémoire
ne supprime pas automatiquement ce coût.

## Recherche et décisions

Deux états de sd.cpp ont été distingués : release publiée `master-841-6b3edaa`
et code amont récupéré au commit `d8fb10c02977c8ca999f3fb4e02df9ecf10f7ba6`.
Le numéro de release ne suffit pas à déduire toutes les options : le programme
interroge le binaire effectivement installé. Aucun binaire expérimental n'est
imposé à l'installation.

- [Placement mémoire sd.cpp](https://github.com/leejet/stable-diffusion.cpp/blob/d8fb10c02977c8ca999f3fb4e02df9ecf10f7ba6/docs/backend.md) : le nouvel auto-fit choisit un GPU de calcul et répartit la résidence des poids. Ce n'est pas une addition automatique de la puissance de toutes les cartes.
- [Gestion mémoire et préchargement](https://github.com/leejet/stable-diffusion.cpp/blob/d8fb10c02977c8ca999f3fb4e02df9ecf10f7ba6/docs/performance.md) : les versions récentes découpent automatiquement les graphes et préchargent les segments. La compatibilité avec le streaming explicite des versions antérieures reste utile.
- [API native du serveur](https://github.com/leejet/stable-diffusion.cpp/blob/master-841-6b3edaa/examples/server/api.md) et [parseur réel](https://github.com/leejet/stable-diffusion.cpp/blob/master-841-6b3edaa/examples/common/common.cpp) : sources de la correction CFG et des images de référence. [Capacités d'annulation](https://github.com/leejet/stable-diffusion.cpp/blob/master-841-6b3edaa/examples/server/routes_sdcpp.cpp) : `cancel_generating` vaut faux.
- [Caches natifs](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/caching.md) : EasyCache, DBCache, TaylorSeer, cache-dit et Spectrum sont déjà exposés. Ils ne sont pas activés arbitrairement sur les modèles distillés à quatre/huit pas ; leur compromis vitesse/qualité doit être mesuré.
- [Téléchargements Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/download) : conservation de `hf_hub_download` et de Xet déjà installé. Le chemin local évite le réseau pour un composant déjà disponible. Aucun mode agressif de transfert n'est imposé au disque.
- [Nunchaku/SVDQuant](https://github.com/nunchaku-tech/nunchaku) et [SageAttention](https://github.com/thu-ml/SageAttention) : solutions intéressantes, mais pas des drapeaux ajoutables à la pile GGML actuelle. Décision d'architecture : ne pas ajouter un second moteur avec ses formats, sa maintenance et ses dépendances sans preuve d'un gain pour les modèles utilisés.
- [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) : les optimisations par offload et échange de blocs appartiennent à son exécution PyTorch. L'intégration optionnelle existante est conservée ; aucune promesse de compatibilité CUDA ou de gain sur la 3060 n'est déduite de benchmarks d'autres cartes.

## Mesures et tests

Les résultats bruts du microbenchmark sont dans `catalog-benchmark.json`.
Pour 200 lectures du catalogue, médiane de trois séries dans cet environnement :

| Implémentation | Temps |
|---|---:|
| Lecture et analyse YAML à chaque appel | 1,586 s |
| Cache, contrôle du fichier et copie indépendante | 0,0158 s |

Soit environ **100 fois moins de temps pour cette opération précise**. Ce chiffre
ne mesure ni le démarrage complet, ni l'échantillonnage GPU, ni le temps total
de création d'une image. Double-cliquer `benchmark-catalog.bat` sous Windows,
ou lancer `./benchmark-catalog.sh` sous Linux/macOS pour reproduire la mesure.

Validation effectuée sous Linux / Python 3.12 avec Gradio 6.17.3,
huggingface_hub 0.36.2, Pillow 12.3.0 et PyYAML 6.0.3 :

- Suite complète : 424 tests exécutés, succès, un ignoré (`psd-tools` absent).
- Serveur Gradio effectivement lancé sur 127.0.0.1 : page et configuration HTTP 200,
  marque Turboslop 5000 présente, 873 composants construits.
- Aller-retour HTTP local avec le schéma natif : CFG, référence encodée et PNG de sortie.
- Binaires Linux officiels sd-cli et sd-server de la release 841 exécutés avec `-h` ;
  détection de 164 et 149 options respectivement, syntaxe historique confirmée.
- Syntaxe Python et cohérence du diff vérifiées.

Le GPU, CUDA, les poids de génération et Windows ne sont pas disponibles pour
une validation de production ici. Aucun gain d'inférence chiffré n'est annoncé.
Sur la machine cible, lancer le benchmark de placement, comparer une même graine
avec/sans moteur résident et vérifier visuellement une édition par références.
La RTX 3060 12 Go reste le choix prudent pour la génération ; la 1080 Ti sur PCIe
x4 n'est pas forcée dans le calcul d'encodage ni dans le partage de diffusion.

## Installation et dépôt GitHub

Extraire la copie dans un nouveau dossier. Sous Windows, lancer `install.bat`,
puis `run.bat`. Les moteurs et les modèles ne sont pas inclus dans l'archive.
Pour réutiliser les modèles de 3000 sans les dupliquer, choisir leur dossier
existant dans les réglages de stockage de 5000, puis redémarrer.

Le dépôt de cette édition est [KSCorpr/Turboslop-5000](https://github.com/KSCorpr/Turboslop-5000).
Les mises à jour de l'application ciblent ce dépôt. Le dépôt 3000 reste intact.
Le commit d'origine est indiqué en tête de ce rapport ; l'archive livrée dans
la conversation contient également un bundle Git complet de l'historique initial.

Pour récupérer cette édition avec Git :

```bash
git clone https://github.com/KSCorpr/Turboslop-5000.git
cd Turboslop-5000
```

Sous Windows, lancer ensuite `install.bat`, puis `run.bat`.
