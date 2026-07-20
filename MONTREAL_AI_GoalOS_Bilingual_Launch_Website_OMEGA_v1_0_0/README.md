# MONTREAL.AI × GoalOS — Site public bilingue / Bilingual public website

## Français

Site statique français-primaire avec miroir anglais complet. Il lance GoalOS comme une institution intégrée de missions, preuve, capacité, identité et capital. Le site public n’utilise ni compte, ni analyse comportementale, ni formulaire serveur, ni portefeuille, ni paiement, ni dépendance d’exécution externe.

### Déploiement sûr

Ce paquet est une **surcouche** destinée au dépôt `MontrealAI/MontrealAI.github.io`, branche `master`, racine du dépôt. Il remplace le fichier racine `index.html` et ajoute des chemins `goalos-*` et autres chemins GoalOS sans supprimer les anciens répertoires.

1. Créer une branche de revue.
2. Conserver le précédent `index.html` dans l’historique Git.
3. Extraire la surcouche à la racine du dépôt.
4. Examiner `SITE_OVERLAY_MANIFEST.json` et le diff.
5. Exécuter `python scripts/verify_site.py` et `node --check goalos-assets/js/site.js`.
6. Fusionner vers `master` seulement après revue.
7. Tester le français, l’anglais, le centre juridique, les téléchargements, le mobile et les anciennes URL.

### Frontière juridique

Aucun paquet ne peut garantir une immunité juridique ou réglementaire. Des conseillers indépendants au Québec, au Canada et dans les juridictions visées doivent examiner l’entité opératrice, les coordonnées, contrats, services réels, données, systèmes et communications avant activation commerciale ou réglementée.

---

## English

French-primary static site with a complete English mirror. It launches GoalOS as one integrated institution for missions, proof, capability, identity and capital. The public site uses no account, behavioural analytics, server form, wallet, payment or external runtime dependency.

### Safe deployment

This package is an **overlay** for `MontrealAI/MontrealAI.github.io`, branch `master`, repository root. It replaces the root `index.html` and adds GoalOS-specific paths without deleting existing legacy directories.

1. Create a review branch.
2. Preserve the previous `index.html` in Git history.
3. Extract the overlay at repository root.
4. Inspect `SITE_OVERLAY_MANIFEST.json` and the diff.
5. Run `python scripts/verify_site.py` and `node --check goalos-assets/js/site.js`.
6. Merge to `master` only after review.
7. Test French, English, legal, downloads, mobile and legacy URLs.

### Legal boundary

No package can guarantee legal or regulatory immunity. Independent Quebec, Canadian and relevant international counsel should review the live operator, entity details, contracts, actual services, data flows, systems and communications before commercial or regulated activation.
