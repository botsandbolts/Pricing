"""Model-faithful XAI accounting tests, including real Captum gradients.

Uses tiny synthetic torch networks and deterministic mock pricing engines,
NEVER downloads BGE or trains price weights. Native XGBoost production path
requires the separately pinned serving-parity checks.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from pricing import neural_price_xai as xai


def vehicle():
    return {
        "year":2015,"manufacturer":"Toyota","model":"Camry SE",
        "odometer":103000.,"type":"sedan","fuel":"gas",
        "transmission":"automatic","drive":"fwd",
        "cylinders":"4 cylinders","condition":"good",
        "title_status":"clean","state":"ca","paint_color":"silver",
        "auction_date":"2015-01-15","auction_grade":"3",
        "listing_text":"Well maintained. New tires. Minor scratches.",
    }


def synthetic(row):
    # Has both main effects and an interaction, so independent ablation cannot
    # pretend to be an additive explanation.
    dollars=11000.
    dollars+=500 if str(row["manufacturer"]).lower()=="toyota" else -180
    dollars+=1200 if row["model"]=="Camry SE" else 0
    dollars+=.02*(110000.-float(row["odometer"]))
    dollars+=1000 if row["title_status"]=="clean" else -2500
    dollars+=200 if row["state"]=="ca" else 0
    dollars+=300 if "tires" in row.get("listing_text","") else 0
    if row["title_status"]=="clean" and float(row["odometer"])<108000:
        dollars+=600
    return dollars


def test_exact_exhaustive_grouped_shapley_and_interaction_allocation():
    actual=vehicle()
    reference=xai.benchmark_reference(actual,"ask","benchmark")
    expected=synthetic(actual)
    r=xai.exact_group_shapley(synthetic,actual,reference,
                             xai.ASK_GROUPS,expected)
    assert r["coalitions_evaluated"]==64
    assert r["target_usd"]==expected
    assert r["base_usd"]==synthetic(reference)
    assert abs(r["reconciliation_error_usd"])<1e-7
    assert sum(v["dollars"] for v in r["groups"])==pytest.approx(
        expected-synthetic(reference),abs=1e-7)
    assert any(v["name"]=="Make, model & year" for v in r["groups"])
    assert r["causal"] is False


def test_shapley_fail_closed_on_receipt_mismatch_or_duplicate_group():
    a=vehicle();ref=xai.benchmark_reference(a,"ask","benchmark")
    with pytest.raises(ValueError,match="Saved price"):
        xai.exact_group_shapley(synthetic,a,ref,xai.ASK_GROUPS,
                                synthetic(a)+200)
    groups=(("Mileage",("odometer",)),("More mileage",("odometer",)))
    with pytest.raises(ValueError,match="two explanation groups"):
        xai.exact_group_shapley(synthetic,a,ref,groups,synthetic(a))
    with pytest.raises(ValueError,match="budget"):
        xai.exact_group_shapley(synthetic,a,ref,
            tuple((f"Group {i}",("year",)) for i in range(8)),
            synthetic(a))
    with pytest.raises(ValueError,match="invalid"):
        xai.exact_group_shapley(lambda row:float("nan"),a,ref,
                                xai.ASK_GROUPS,synthetic(a))


class FakeAsk:
    def __init__(self):
        self.prepare_calls=0
    def prepare_text(self,s):
        self.prepare_calls+=1
        return {"text":str(s)}
    def predict_prepared(self,fields,prepared):
        return {"price":synthetic({**fields,"listing_text":prepared["text"]})}
    def predict(self,description,fields):
        return self.predict_prepared(fields,self.prepare_text(description))


class FakeBid:
    def predict(self,fields):
        return {"price":synthetic({**fields,"listing_text":""})}


def test_ask_uses_two_text_encodes_not_64_and_matches_saved_real_input():
    model=FakeAsk()
    a=vehicle()
    price=model.predict(a["listing_text"],a)["price"]
    model.prepare_calls=0
    r=xai.explain_ask(model,a,a["listing_text"],price,"benchmark")
    assert r["coalitions_evaluated"]==64
    assert r["text_encoder_runs"]==2
    assert model.prepare_calls==2
    assert abs(r["reconciliation_error_usd"])<.00001


def test_bid_64_predictions_reconcile_without_claiming_sale_profit():
    a=vehicle();model=FakeBid()
    price=model.predict(a)["price"]
    r=xai.explain_bid(model,a,price,"benchmark")
    assert r["market"]=="bid"
    assert r["coalitions_evaluated"]==64
    assert abs(r["reconciliation_error_usd"])<1e-6
    assert not r["causal"]


def test_date_transport_is_algebraic_not_made_up_shap():
    receipt={"age_factor":.84,"market_factor":1.16,
             "price":10000*.84*1.16,"date":"2020-10-01"}
    r=xai.historical_date_bridge(10000,receipt)
    assert r["vehicle_age_effect_usd"]==-1600
    assert r["market_index_effect_usd"]==pytest.approx(1344)
    assert sum(r[k] for k in (
        "historical_anchor_usd","vehicle_age_effect_usd",
        "market_index_effect_usd"))==pytest.approx(r["date_adjusted_usd"])


def test_text_phrase_removal_is_actual_rerun_and_not_additive_shap():
    model=FakeAsk();a=vehicle();text=a["listing_text"]
    price=model.predict(text,a)["price"]
    r=xai.phrase_removal_effects(model,a,text,price,6)
    assert len(r["items"])>=2
    assert r["encoder_runs"]==len(r["items"])+1
    assert not r["causal"]
    assert "DO NOT add" in r["warning"]
    with pytest.raises(ValueError,match="does not match"):
        xai.phrase_removal_effects(model,a,text,price+100,6)


class TinyAskTorch(torch.nn.Module):
    def forward(self,num,cats,text):
        price=10000.+100*num[:,0]+200*text[:,0]
        log=torch.log(price)
        return torch.stack((log,log+0.03),dim=1)


class FakeCaptumAsk:
    def __init__(self):
        self.price_network=TinyAskTorch()
    def prepare_text(self,s):
        vec=np.zeros((1,384),dtype=np.float32)
        vec[0,0]=1. if "tires" in s else .2
        return {"embedding":vec}
    def encode_inputs(self,fields,prepared):
        nums=np.zeros((1,7),dtype=np.float32)
        nums[0,0]=float(fields["odometer"])/10000.
        cats=np.ones((1,11),dtype=np.int64)
        return nums,cats,prepared["embedding"]


class TinyBidTorch(torch.nn.Module):
    def forward(self,numbers,categories):
        return torch.log(8000.+100*numbers[:,0])


class BidFeatures:
    def features(self,fields):
        numeric=np.zeros((1,5),dtype=np.float32)
        numeric[0,0]=float(fields["odometer"])/10000.
        return numeric,np.ones((1,10),dtype=np.int64),numeric[0],[]


class FakeCaptumBid:
    def __init__(self):
        self.original=BidFeatures()
        self.model=TinyBidTorch()


def test_captum_ask_continuous_dollars_are_complete():
    pytest.importorskip("captum")
    a=vehicle()
    r=xai.captum_integrated_gradients(
        FakeCaptumAsk(),"ask",a,
        {**a,"odometer":95000.,
         "listing_text":"A used car."},
        a["listing_text"],steps=16)
    assert len(r["attributions"])==8
    assert r["fixed_categorical_inputs"] is True
    assert abs(r["reference_branch_usd"]+
               sum(p["dollars"] for p in r["attributions"])-
               r["price_branch_usd"])<5.
    assert r["price_branch_usd"]>r["reference_branch_usd"]


def test_captum_bid_does_not_claim_to_cover_xgboost():
    pytest.importorskip("captum")
    a=vehicle()
    r=xai.captum_integrated_gradients(
        FakeCaptumBid(),"bid",a,
        {**a,"odometer":95000.},steps=16)
    assert len(r["attributions"])==5
    assert "neural branch" in r["scope"].lower()
    assert abs(r["convergence_delta_usd"])<5.


def test_real_xgboost_native_treeshap_log_and_dollar_reconciliation():
    xgb=pytest.importorskip("xgboost")
    from pricing.neural_bid_blend_live import CATS
    import numpy as np
    names=["age","log_miles","auction_grade",*CATS[:8]]
    types=["q","q","q",*(["c"]*8)]
    rng=np.random.default_rng(20261107)
    raw=np.column_stack([
        rng.uniform(1,20,size=80),
        rng.uniform(8,13,size=80),
        rng.uniform(1,5,size=80),
        rng.integers(0,3,size=(80,8)),
    ]).astype(np.float32)
    targets=np.log(7000.+400*raw[:,0]+100*raw[:,1]+1100*(raw[:,3]==1))
    matrix=xgb.DMatrix(raw,label=targets,
                       feature_names=names,feature_types=types,
                       enable_categorical=True)
    booster=xgb.train({"objective":"reg:squarederror","tree_method":"hist",
                       "max_depth":2,"eta":.3,"nthread":2},matrix,num_boost_round=8)
    schema={name:["unknown","toyota","ford","other"] for name in CATS[:8]}

    class LightweightOriginal:
        xgb=booster
        xgb_schema=schema
        def features(self,fields):
            year=int(fields["year"])
            age=float(2015-year)
            odom=float(fields["odometer"])
            grade=float(fields.get("auction_grade",3))
            r=np.array([year,age,np.log1p(odom),
                        np.log1p(odom/max(age,1.)),grade],dtype=float)
            return (np.zeros((1,5),dtype=np.float32),
                    np.ones((1,10),dtype=np.int64),r,[])

    class Published:
        original=LightweightOriginal()

    from pricing.hierarchy_sparse_live import design_matrix
    fields=vehicle()
    fields.update(manufacturer="toyota",model="corolla",
                  title_status="unknown",state="ca")
    pack={k:fields.get(k) for k in CATS[:8]}
    pack.update(odometer=fields["odometer"],
                auction_grade=float(fields["auction_grade"]))
    matrix,_=design_matrix(booster,schema,None,pack,0.)
    actual_xgb=float(np.exp(booster.predict(matrix,output_margin=True)[0]))
    result=xai.bid_treeshap(Published(),fields,actual_xgb)
    assert result["tree_prediction_usd"]==pytest.approx(actual_xgb,abs=.5)
    assert len(result["features"])==len(names)
    assert abs(result["log_additivity_error"])<.002
    assert abs(result["dollar_reconciliation_error"])<.15
    with pytest.raises(ValueError,match="displayed Lemon XGBoost"):
        xai.bid_treeshap(Published(),fields,actual_xgb+100)
