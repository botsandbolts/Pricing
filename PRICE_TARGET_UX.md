# Two prices, two different questions

The calculator now defines the two economic targets before displaying any prices:

- **ASK · Seller's advertised price:** what a seller might put in an advert, before negotiation. The source is the Craigslist Used Cars 2021 listing snapshot. It is not a completed retail sale.
- **AUCTION · Wholesale sale estimate:** what a wholesale auction sale might achieve. The source is Used Car Auction Prices (Kaggle / tunguz), with reported historical sales in 2014–2015. It is not a live bid, a dealer offer, or the Cars & Bids website.

Purple and green cards supplement explicit text labels; colour is not the only distinction. The explanation stays visible on mobile, with cards stacking vertically. The gap warning is visible without opening a technical panel.

Every served price keeps its existing numeric value and has a nearby date/basis label. The main calculator displays the May 2021 asking anchor and May 2021 transported auction equivalent. The two-market calculators warn when the selected auction month differs from the asking anchor. The Value over time panel explicitly calls both date-adjusted estimates **scenarios**, including past dates; it does not certify current transaction accuracy. The description summary follows both main price cards, so it cannot separate their comparison.

The gap is not dealer profit, a validated retail-sale conversion, a prediction interval, or guaranteed negotiating room. These sources do not provide paired wholesale purchase and final retail sale outcomes for the same vehicles. Fees, reconditioning, transport, holding costs and other expenses would also be needed to calculate a margin. Even aligning dates does not resolve market selection and calibration differences.

`pricing/price_meaning.py` centralizes the definitions and result presentation. The main calculator, default and legacy two-market calculators, time scenarios and Deal Detective comparison use these definitions. Internal market IDs, analytics IDs, exported historical schemas, models, weights, targets, predictions and uncertainty methods are unchanged.
