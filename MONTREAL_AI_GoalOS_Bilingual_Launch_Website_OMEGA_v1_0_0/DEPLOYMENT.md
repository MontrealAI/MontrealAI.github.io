# Déploiement et retour arrière / Deployment and rollback

## Français

### Cible

- Dépôt : `MontrealAI/MontrealAI.github.io`
- Branche : `master`
- Emplacement : racine du dépôt
- Publication : configuration GitHub Pages existante, branche/racine

### Méthode recommandée

1. Créer `goalos-bilingual-launch-v1` depuis le dernier `master`.
2. Vérifier que la copie locale est propre et sauvegardée.
3. Extraire le ZIP **DEPLOY OVERLAY** à la racine, sans supprimer les chemins non listés.
4. Examiner le diff, surtout `index.html`, `.github/`, `goalos-*`, `en/`, `goalos-assets/` et `goalos-documents/`.
5. Exécuter les contrôles inclus.
6. Prévisualiser par HTTPS sur une branche ou un environnement de revue.
7. Obtenir la revue juridique et de sécurité appropriée.
8. Fusionner puis effectuer la recette en direct.

### Retour arrière

Revenir au commit précédent ou restaurer l’ancien `index.html` depuis Git. La surcouche n’est pas conçue pour supprimer les anciennes URL.

### Limite GitHub Pages

GitHub Pages ne permet pas de définir tous les en-têtes HTTP personnalisés. Le site contient une CSP méta restrictive. Un hébergement à assurance supérieure devrait ajouter au niveau serveur : HSTS, CSP complète, Permissions-Policy, Referrer-Policy, X-Content-Type-Options et, lorsque pertinent, protections contre l’encadrement.

---

## English

### Target

- Repository: `MontrealAI/MontrealAI.github.io`
- Branch: `master`
- Location: repository root
- Publication: existing GitHub Pages branch/root configuration

### Recommended method

1. Create `goalos-bilingual-launch-v1` from current `master`.
2. Confirm the local copy is clean and backed up.
3. Extract the **DEPLOY OVERLAY** ZIP at repository root without deleting unlisted paths.
4. Inspect the diff, especially `index.html`, `.github/`, `goalos-*`, `en/`, `goalos-assets/` and `goalos-documents/`.
5. Run the included verification gates.
6. Preview over HTTPS on a branch or review environment.
7. Obtain appropriate legal and security review.
8. Merge and perform live acceptance.

### Rollback

Revert the launch commit or restore the previous root `index.html` from Git. The overlay is not designed to delete legacy URLs.

### GitHub Pages limitation

GitHub Pages does not support every custom HTTP response header. The site therefore includes a restrictive meta CSP. Higher-assurance hosting should add server-level HSTS, full CSP, Permissions-Policy, Referrer-Policy, X-Content-Type-Options and framing protections where appropriate.
