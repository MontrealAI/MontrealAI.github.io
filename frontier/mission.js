(function(){
"use strict";
const fr=document.documentElement.lang.startsWith("fr");
const t=fr?{
need:"Décrivez d'abord un résultat concret.",ready:"Note de mission préparée localement. Aucun message n'a été envoyé.",copied:"Note copiée.",copyfail:"La copie automatique a échoué. Sélectionnez et copiez le texte ci-dessous.",
heading:"MONTREAL.AI — MANDAT FRONTIÈRE Ω",outcome:"Résultat souhaité",obstacle:"Contrainte ou approche actuelle",accept:"Preuve d'acceptation",na:"À définir avec le commanditaire",
control:"Contrôle crédible",controlbody:"Comparer l'option la plus compétente déjà disponible : processus humain avec logiciels habituels, fournisseur existant, outils IA actuels, partenariat ou absence d'achat.",
discovery:"Découverte et invention",discoverybody:"Chercher des transferts structuraux, des contre-exemples, des stratégies nouvelles, des changements de contraintes et des instruments de mesure appropriés.",
proof:"Preuve indépendante",proofbody:"Définir préalablement un critère observable; consigner les tentatives, les risques, les coûts complets et les contrôles; faire vérifier le résultat par une partie distincte de son constructeur.",
inherit:"Héritage et économie",inheritbody:"Conserver uniquement les méthodes et capacités autorisées, utiles et validées. Tester sur une nouvelle mission leur avantage net après adaptation et entretien.",
boundary:"Statut",boundarybody:"Hypothèse commerciale et protocole d'investigation. Aucune réalisation, aucun gain et aucune approbation ne sont encore revendiqués.",
emailSubject:"MONTREAL.AI — Proposition de mandat frontière"
}:{
need:"Start with one concrete outcome.",ready:"Mission brief prepared locally. No message has been sent.",copied:"Brief copied.",copyfail:"Automatic copy was unavailable. Select and copy the text below.",
heading:"MONTREAL.AI — FRONTIER MANDATE Ω",outcome:"Desired outcome",obstacle:"Constraint or current approach",accept:"Acceptance evidence",na:"To be defined with the sponsor",
control:"Credible control",controlbody:"Compare the strongest available complete route: capable humans using conventional software, an incumbent provider, existing AI tooling, a partnership, or no purchase.",
discovery:"Discovery and invention",discoverybody:"Investigate structural transfers, counterexamples, new strategies, transformed constraints, and the measurement instruments needed to distinguish them.",
proof:"Independent proof",proofbody:"Preregister an observable acceptance test; record attempts, risks, full costs and controls; use a reviewer independent from construction.",
inherit:"Capability inheritance and economics",inheritbody:"Retain only authorized, useful, verified methods and capabilities. On a fresh mission, test net advantage after adaptation and maintenance.",
boundary:"Claim boundary",boundarybody:"Commercial inquiry and investigation protocol only. No delivered outcome, financial benefit, or acceptance is claimed yet.",
emailSubject:"MONTREAL.AI — Frontier mandate inquiry"
};
const form=document.getElementById("mission-form"),result=document.getElementById("brief"),status=document.getElementById("status");
const copy=document.getElementById("copy"),email=document.getElementById("email");
let brief="";
const val=(id)=>document.getElementById(id).value.trim();
function line(a,b){return a+": "+(b||t.na);}
function generate(){
 const goal=val("goal");if(!goal){status.textContent=t.need;document.getElementById("goal").focus();return;}
 brief=[t.heading,"",line(t.outcome,goal),line(t.obstacle,val("barrier")),line(t.accept,val("measure")),"",
 "01. "+t.control+" — "+t.controlbody,
 "02. "+t.discovery+" — "+t.discoverybody,
 "03. "+t.proof+" — "+t.proofbody,
 "04. "+t.inherit+" — "+t.inheritbody,
 "",line(t.boundary,t.boundarybody),"","info@montreal.ai"].join("\n");
 result.textContent=brief;result.closest("[data-output]").hidden=false;
 status.textContent=t.ready;copy.disabled=false;email.disabled=false;
}
form.addEventListener("submit",function(e){e.preventDefault();generate();});
copy.addEventListener("click",async function(){if(!brief)return;try{await navigator.clipboard.writeText(brief);status.textContent=t.copied;}catch(e){status.textContent=t.copyfail;}});
email.addEventListener("click",function(){if(!brief)return;const subject=encodeURIComponent(t.emailSubject),body=encodeURIComponent(brief);window.location.href="mailto:info@montreal.ai?subject="+subject+"&body="+body;});
})();
