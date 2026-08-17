'use strict';
self.onmessage=e=>{
  const {type,id,observations=[]}=e.data||{};if(type!=='act')return;
  const actions=observations.map(o=>{
    const risk=Math.max(0,Math.min(1,o.stormRisk));
    const reserve=Math.max(0.28,0.35+0.5*risk);
    const available=Math.max(0,o.stateOfCharge-reserve);
    const priceSignal=(o.price-50)/70;
    const discharge=Math.max(0,Math.min(available,priceSignal>0?available*Math.min(1,priceSignal):0));
    const charge=Math.max(0,Math.min(1-o.stateOfCharge,o.price<35?(35-o.price)/50:0));
    return {reserveTarget:reserve,discharge,charge,escalate:risk>0.88&&o.stateOfCharge<0.55};
  });
  self.postMessage({type:'actions',id,actions});
};
