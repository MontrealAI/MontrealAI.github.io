(() => {
  const form = document.querySelector('[data-forecast-form]');
  if (!form) return;
  const fr = document.documentElement.lang.toLowerCase().startsWith('fr');
  const output = document.querySelector('[data-forecast-output]');
  const pre = document.querySelector('[data-forecast-json]');
  const download = document.querySelector('[data-forecast-download]');
  const copy = document.querySelector('[data-forecast-copy]');
  let latest = null;
  const isoToday = new Date().toISOString().slice(0,10);
  const cutoff = form.querySelector('[name="cutoff"]');
  if (cutoff && !cutoff.value) cutoff.value = isoToday;
  const labels = fr ? {
    title:'Dossier de prévision consciente de l’accélération Ω',
    generated:'généré_localement', target:'question_ou_cible', horizon:'horizon', cutoff:'date_limite_des_preuves', notes:'contraintes_et_contexte',
    required:['vérifier les sources primaires les plus récentes','mesurer le rythme par rapport aux 6–12 mois précédents','ajuster des modèles linéaire, exponentiel, accélérant, contraint et à changement de régime','produire des scénarios de base, accéléré et discontinu','identifier les goulots d’autorité, confiance, adoption, réglementation, capital, énergie, chaîne d’approvisionnement et exécution physique','définir des indicateurs déclencheurs, le plan optimal actuel et un point de recalcul daté'],
    analysisKey:'analyse_requise', scenariosKey:'scénarios', scenarios:['de_base','accéléré','discontinu'], notice:'Dossier local de cadrage. Ce fichier n’est ni une prévision achevée, ni un conseil professionnel, ni une garantie.'
  } : {
    title:'Acceleration-Aware Forecast Ω Brief',
    generated:'generated_locally', target:'question_or_target', horizon:'horizon', cutoff:'evidence_cutoff', notes:'constraints_and_context',
    required:['verify the newest primary sources','measure current pace against 6–12 months earlier','fit linear, exponential, accelerating, constrained and regime-change models','produce base, accelerated and discontinuous scenarios','identify authority, trust, adoption, regulation, capital, energy, supply-chain and physical-execution bottlenecks','define trigger indicators, the optimal current plan and a dated recalculation point'],
    analysisKey:'required_analysis', scenariosKey:'scenarios', scenarios:['base','accelerated','discontinuous'], notice:'Browser-local framing brief. This file is not a completed forecast, professional advice or a guarantee.'
  };
  const build = () => {
    const data = new FormData(form);
    const obj = {
      instrument: labels.title,
      version: '1.0.0',
      [labels.generated]: new Date().toISOString(),
      [labels.target]: String(data.get('target') || '').trim(),
      [labels.horizon]: String(data.get('horizon') || '').trim(),
      [labels.cutoff]: String(data.get('cutoff') || '').trim() || null,
      [labels.notes]: String(data.get('notes') || '').trim() || null,
      [labels.analysisKey]: labels.required,
      [labels.scenariosKey]: labels.scenarios,
      notice: labels.notice
    };
    return obj;
  };
  form.addEventListener('submit', e => {
    e.preventDefault();
    if (!form.reportValidity()) return;
    latest = build();
    pre.textContent = JSON.stringify(latest, null, 2);
    output.classList.add('open');
    output.scrollIntoView({behavior:'smooth', block:'nearest'});
  });
  download?.addEventListener('click', () => {
    if (!latest) return;
    const blob = new Blob([JSON.stringify(latest,null,2)+'\n'], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href=url; a.download=fr?'dossier-prevision-acceleration.json':'acceleration-aware-forecast-brief.json';
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  });
  copy?.addEventListener('click', async () => {
    if (!latest) return;
    const text=JSON.stringify(latest,null,2);
    try { await navigator.clipboard.writeText(text); copy.textContent=fr?'Copié':'Copied'; }
    catch { const ta=document.createElement('textarea'); ta.value=text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); copy.textContent=fr?'Copié':'Copied'; }
    setTimeout(()=>copy.textContent=fr?'Copier':'Copy',1200);
  });
})();
