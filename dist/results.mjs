const dollars=n=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(n);
try{
 const response=await fetch('/tabular-huber-25.json');if(!response.ok)throw Error('not finished');
 const report=await response.json();
 const champion=report.experiments.find(r=>r.id===report.selected_on_validation.mae_usd);
 document.getElementById('baseline-summary').textContent=`${report.experiments.length} completed tabular-only configurations. Validation-MAE winner: ${champion.id}. Held-out test MAE ${dollars(champion.test.mae_usd)}; RMSE ${dollars(champion.test.rmse_usd)}. E5 stack not yet evaluated.`;
 const render=()=>{
  const body=document.getElementById('baseline-rows');body.replaceChildren();
  const rate=document.getElementById('rate-filter').value;
  for(const r of [...report.experiments].filter(r=>rate==='all'||r.removed_fraction===Number(rate)).sort((a,b)=>a.validation.mae_usd-b.validation.mae_usd)){
   const tr=document.createElement('tr');
   for(const value of [r.id,dollars(r.validation.mae_usd),dollars(r.validation.rmse_usd),dollars(r.test.mae_usd),dollars(r.test.rmse_usd)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}
   body.append(tr);
  }
 };
 document.getElementById('rate-filter').addEventListener('change',render);render();
}catch{document.getElementById('baseline-summary').textContent='No completed baseline report is available yet.';}
