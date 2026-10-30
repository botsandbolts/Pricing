"""A deterministic cross-panel fact report; BART only condenses this report."""
import json
from pathlib import Path
from types import SimpleNamespace


def _read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def extend_report(report, root, fields, base_result, receipts, transport, ask_folder, bid_folder):
    """Use the same car/anchors as the receipt, never another user's session data.

    Historical values are reconstructions at constant entered mileage, not sales.
    No extra MiniLM inference, expensive SHAP fit or timeline build is triggered.
    """
    root=Path(root);out=dict(report);sections=[]
    sections.append({'title':'Price targets & datasets','text':
        'The asking model learns advertised Craigslist Used Cars prices from April and May 2021. '
        'The auction model learns platform-reported wholesale selling prices from the Used Car Auction Prices dataset (Kaggle / tunguz), covering 2014 and 2015. '
        'The auction dataset is not the Cars & Bids enthusiast-auction website. An asking price is a seller request; an auction estimate describes a different market, not a guaranteed offer.'})
    if base_result.get('predicted_text_price') is not None:
        sections.append({'title':'Description branch','text':f"MiniLM's description-only price feature is {base_result['predicted_text_price']:,.2f} dollars in its historical training-price units. It enters the asking CatBoost model alongside vehicle facts; the auction data contain no descriptions."})
    explanation=base_result.get('explanation',{})
    drivers=explanation.get('features',[])
    if drivers:
        ordered=sorted(drivers,key=lambda r:abs(r['contribution']),reverse=True)
        sections.append({'title':'Feature contributions','text':
            'The historical asking model uses native TreeSHAP. Contributions below are log-price margin units, not dollars or causal effects. '+
            '; '.join(f"{r['feature']}={r.get('value')}: {r['contribution']:+.4f}" for r in ordered)+
            '. The visible time-evidence waterfalls separately explain the dollar movement from the historical anchor to the selected date. Phrase explanations, when requested, explain the description model response; they are not evidence that a word changes a real sale price.'})
    historical=[]
    if transport is not None:
        import pandas as pd
        from marketbridge.time_engine import today
        target=pd.Timestamp(report['valuation_date']);dates=[pd.Timestamp('2021-05-01')]
        dates += [pd.Timestamp(f'{y}-01-01') for y in range(2022,target.year+1)]
        dates += [target]
        for side,receipt in receipts.items():
            class Anchor:
                manifest={'feature_names':['age']}
                def predict(self,text,car):return {'price':receipt['anchor_price_reference_usd']}
            for date in sorted(set(dates)):
                if date>target:continue
                try:
                    r=transport.estimate(Anchor(),'',fields,date,side,
                        receipt.get('generation_analogue',False),as_of=today(),
                        anchor_price=receipt['anchor_price_reference_usd'] if side=='ask' else None)
                    historical.append({'market':side,'date':str(date.date()),'price_usd':round(r['price'],2),
                                       'mileage':fields['odometer'],'index_status':r['market'].get('kind')})
                except ValueError:continue
        sections.append({'title':'Historical prices & value over time','text':
            'At the entered mileage held constant, reconstructed same-car scenarios are: '+
            '; '.join(f"{r['date']} {r['market']} {r['price_usd']:,.2f} dollars" for r in historical)+
            '. These are model-derived historical scenarios, not observed sales of this car. The Value over time panel offers monthly detail and an alternative annual-driving assumption; changing mileage changes that scenario. Solid lines are reconstructed past values and dashed lines are future scenarios.'})
        for side in ['ask','bid']:
            curve=_read(root/f'models/time-transport/{side}-evaluation.json')
            if curve:
                scores=curve.get('assessment',{}).get('test',{})
                sections.append({'title':side.title()+' depreciation evidence','text':
                    'The selected depreciation component is '+str(curve.get('selected'))+'. '+
                    (f"Its historical component test MAE is {scores['mae_usd']:,.2f} dollars on {scores['rows']:,} cars. " if scores.get('mae_usd') is not None else '')+
                    'Component validation does not validate the whole current-dollar car estimate.'})
        forecast=_read(root/'models/time-transport/market-backtest.json')
        if forecast:
            selected=forecast.get('selected')
            scores=[r for r in forecast.get('summary',[]) if r.get('method')==selected and r.get('stage')=='locked_test']
            sections.append({'title':'Market clock backtests','text':
                'The selected monthly market forecast method is '+str(selected).replace('_',' ')+'. Rolling-origin held-out backtests of this method: '+
                '; '.join(f"{r['horizon']} month horizon, {r['forecasts']} forecasts, mean absolute log-index error {r['mean_abs_log_error']:.4f}" for r in scores)+
                '. Alternative methods and all backtest results are available in the time-evidence panel. The aggregate retail CPI is an assumption when transporting wholesale values; it is not a daily individual-car transaction feed.'})
    for side,folder in [('ask',ask_folder),('bid',bid_folder)]:
        if folder is None:continue
        manifest=_read(Path(folder)/'manifest.json');evaluation=_read(Path(folder)/'evaluation.json')
        text=side.title()+' estimator: '+manifest.get('pipeline','unspecified')+'.'
        cleaning=manifest.get('outlier_filter',{})
        text+=' Training method: '+manifest.get('training_regime','unspecified')+'.'
        if cleaning.get('stage_count')==5:
            text+=f" Five CatBoost residual rounds each remove 5 percent of survivors; {cleaning['removed_rows']:,} of {cleaning['original_rows']:,} eligible cars were removed."
        scores=evaluation.get('assessment',{})
        for name in ['original_unfiltered_test','filtered_test']:
            score=scores.get(name,{})
            if score.get('mae_usd') is not None:
                text+=f" {name.replace('_',' ')}: MAE {score['mae_usd']:,.2f} dollars on {score['rows']:,} historical cars."
        text+=' Filtered validation and test scores are optimistic because outcomes select which cars remain. Original-test diagnostics have informed development; a new external cohort is needed for an unbiased accuracy claim.'
        sections.append({'title':side.title()+' model & evaluation','text':text})
    sections.append({'title':'Other studio tools','text':
        'The Two-market garage compares asking and auction targets. Model lab compares trained alternatives. Deal Detective can inspect a supplied listing and show a model comparison; no listing has been inspected by this valuation unless you explicitly supply one there. Sharing creates a price card and link. Challenge the model compares a user guess with an estimate; it is not an accuracy benchmark.'})
    out['sections']=sections;out['historical_scenarios']=historical
    return out
