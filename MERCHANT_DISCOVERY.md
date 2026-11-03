# Merchant + Meaning discovery
A separate Drive Chemistry engine, enabled only after its verified catalogue is installed.

## Business behavior
Five diverse real cars collect direct reactions. Every liked car supplies a separate semantic interest; dislikes subtract graded evidence instead of vetoing entire clusters. The shortlist is scored highest-first. Rating results and resubmitting refines the session, excluding already-rated cars. A historical asking-price limit is optional and explicit. Price displays are historical and not current valuations.

Merchant groups require an automotive organization mention and one corroborating non-platform domain. Conflicting organization identities remain unresolved. These are inferred candidates, not verified businesses; legal identity, ownership and merchant expertise have not been audited.

The current extraction resolved zero merchant candidates and zero linked adverts, so the inventory branch has insufficient support and recommendations currently use description semantics alone. Inventories can connect through shared vehicle-type attributes rather than isolated advert IDs. This is a hypothesis about commercial assortment, not customer purchase preference. Unknown identities never exclude a car.

## Offline implementation
Recover canonical original descriptions for retained reference rows, checking row IDs and prices. The packaged baseline uses dslim/bert-base-NER; a separate GLiNER extraction job is in progress. That baseline extracts organization mentions. Only automotive-business candidates with a supporting domain are resolved; lenders/platforms are excluded conservatively. Canonicalization ignores punctuation and common company suffixes. The first version does not resolve every address, alias or corporate branch.

Vehicle semantics suppress resolved seller names and contact/financing lines. The default uses all-MiniLM-L6-v2 embeddings of 110,832 retained descriptions, with a 256-token limit and a 96-dimensional truncated SVD projection. Stored document vectors are normalized offline; serving needs no text model. TF-IDF is removed from this engine’s selector and is not a fallback. Doc2Vec is no longer the default.

Merchant inventory counts use shared make, body and make/model columns. Row normalization reduces dealer-size effects; inverse merchant frequency reduces ubiquitous attributes. TruncatedSVD supplies up to 32 inventory-association dimensions. No user feedback labels are invented for this model.

The initial hybrid uses 80% semantic and 20% inventory association where available, with a graded dislike penalty of 35%. Unavailable inventory falls back to semantics. Each liked vehicle contributes separately using its rating strength. Up to 500 candidates enter diversity selection, with at most two exact make/model types. No same-make restriction applies. These weights are declared baselines, not learned optima.

## Verification and limits
Unit checks cover platform/lender ambiguity, preference ordering, explicit budget filtering and exclusion of rated cars. Each initial seed is tested as the sole like against four dislikes and must return ten distinct cars. A real Streamlit five-rating/submit/shortlist-render test gates publication.

There is no independently labelled relevance benchmark or manually audited entity precision yet. Existing canonical preparation has already deduplicated text/VINs, so this first catalogue is not an untouched original-advert archive. Dealer co-stock affinity is not evidence that a shopper wants the same cars. No fraud, transaction or conversion claims are made.

Existing KMeans/HDBSCAN experiments remain available. Activation requires a manifest containing the passed UI check. Serving artifacts are independent; the main pricing models are unchanged.
