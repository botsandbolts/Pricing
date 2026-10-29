export default function(component) {
  const events=component.data.events || [];
  let attempts=0;
  const deliver=()=>{
    const tracker=window.__pricingInteractions;
    if(!window.__pricingGoogleAnalytics) {if(attempts++<20)setTimeout(deliver,250);return;}
    if(!window.__pricingGoogleAnalytics.allowed)return;
    if(!tracker) {if(attempts++<20)setTimeout(deliver,250);return;}
    for(const event of events) {
      if(event.name!=='estimate_completed' || tracker.state.seen.has(event.id))continue;
      const side=event.params?.market_side;
      const feature=event.params?.feature_name;
      if(!['ask','bid'].includes(side) || !['calculator','two_market','value_over_time','model_lab'].includes(feature))continue;
      // Streamlit executes hidden tabs too. Do not count their passive calculations.
      if(feature!==tracker.state.panel)continue;
      // Do not lose a completion just because its panel was hidden at delivery time.
      if(document.hidden)continue;
      tracker.send('estimate_completed',{market_side:side,feature_name:feature});
      tracker.state.seen.add(event.id);
    }
    if(tracker.state.seen.size>2000)tracker.state.seen.clear();
  };
  deliver();
  const onVisible=()=>deliver();
  document.addEventListener('visibilitychange',onVisible);
  return ()=>document.removeEventListener('visibilitychange',onVisible);
}
