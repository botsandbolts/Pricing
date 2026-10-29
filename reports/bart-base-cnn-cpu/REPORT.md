# BART-base-CNN versus SmolLM2-135M: CPU smoke test

BART checkpoint: `ainize/bart-base-cnn` at `b90bc9a7c93de6449a8c531ed5f957d84649b99a`.

| Metric | BART-base-CNN | SmolLM2-135M, structured prompt |
|---|---:|---:|
| Parameters | 139,420,416 | 134,515,008 |
| Cases / generations | 12 / 24 | 12 / 24 |
| Median generation time | 1.45 s | 5.20 s |
| Range | 0.92–4.62 s | 3.98–11.50 s |
| Peak process RSS | 1472 MiB | 1219 MiB |
| EOS-finished outputs | 22/24 | 16/24 |
| Identical repeated pairs | 12/12 | 12/12 |
| Initial load, including download | 40.0 s | 29.3 s |

Both run sequentially on CPU with two PyTorch threads, FP32, greedy decoding and
128-new-token output caps. Same eight synthetic cases and four public examples;
model-appropriate input formatting differs (BART plain story, SmolLM chat template).
Different output lengths contribute to speed differences. This is not a controlled
per-token architecture speed claim. The simpler SmolLM prompt had a 3.37-second
median on only the eight controlled cases, with significant factual/omission issues.

BART is about 3.6 times faster by median in this particular same-case comparison.
This does not establish better general accuracy. Peak RSS includes Python, PyTorch,
weights and inference allocations; it is not incremental app memory. The production
app uses a different PyTorch version and a shared hosting environment. No deployment.

## Manual inspection

- AC fault: preserved the nonworking AC, but duplicated identity and shortened Ford to F.
- Salvage: omitted the crucial salvage title while keeping the reported accident repair.
- Unknown history: retained the uncertainty reasonably.
- Mileage conflict: recognized conflict, but rendered 95,000 as 95,00 in one phrase.
- Oil leak: retained unrepaired leak and brake replacement.
- Battery negation: omitted the unreplaced traction battery/reduced range and rendered 67,000 as 67,00.
- Seller identity: omitted cracked windshield and introduced F-F-150 wording.
- Injected instruction: did not claim inspection, but invented F1 as a model label and omitted the AC fault.
- Public Lexus 0: emphasized dealer disclaimer and added unsupported exclusions wording.
- Public Fiat 25: copied SEO model keywords; both runs exhausted the output budget.
- Public Hyundai 50: repeated conflicting source mileages without explaining the discrepancy.
- Public Lexus 75: repeated ES and added a US-registration claim requiring verification.

These are manual observations from a small nonrandom test, not an accuracy percentage.
Automated keyword and number checks in JSON are diagnostic and can have false positives.
No gold summaries or ROUGE score were created. The result supports BART as a faster
candidate, but not as an unrestricted default without factual validation.

## Every BART output

**ac_fault** — 1.70 seconds; 44 input tokens; source chunks 1

```text
The listed vehicle is 2014 Ford Focus with 95,000 miles .
The listed car is 2014 F Focus with a transmission: automatic .
New tires installed. Air conditioning does not work .
```

**salvage** — 1.38 seconds; 56 input tokens; source chunks 1

```text
The listed vehicle is 2012 Honda Civic with 140,000 miles .
Front-end accident damage was repaired, according to the seller .
No inspection report is available .
```

**unknown_history** — 1.15 seconds; 37 input tokens; source chunks 1

```text
The listed vehicle is 2016 Toyota Camry with 80,000 miles .
Service history is unknown. Transmission type and accident history were not provided .
```

**mileage_conflict** — 1.59 seconds; 54 input tokens; source chunks 1

```text
The listed vehicle is 2014 Ford Focus with 95,000 miles .
The description says the odometer reads 145,000 .
the structured listing says 95,00 .
These values conflict .
```

**oil_leak** — 1.44 seconds; 45 input tokens; source chunks 1

```text
The listed vehicle is 2011 BMW 328i with 130,000 miles .
Oil leak remains unrepaired .
Rear brakes were replaced last month .
The seller says it starts and drives .
```

**battery_negation** — 0.92 seconds; 59 input tokens; source chunks 1

```text
The listed vehicle is 2015 Nissan Leaf with 67,000 miles .
The vehicle is expected to have 67,00 miles .
```

**seller_identity** — 1.17 seconds; 49 input tokens; source chunks 1

```text
The listed vehicle is 2010 Ford F-150 with 170,000 miles .
The listed car is 2010 F-F-150 .
No warranty is offered .
```

**instruction_in_listing** — 1.09 seconds; 53 input tokens; source chunks 1

```text
The listed vehicle is 2014 Ford Focus with 95,000 miles .
The listed car is 2014 F1 with 95k miles .
```

**public_0** — 4.14 seconds; 934 input tokens; source chunks 1

```text
The listed vehicle is 2015 lexus dealer* luxury pkg* rear dvd* preffered pkg*.mark levinson*2-keys*like new with 72,249 miles.
The Dealership makes no representations, expressed or implied, to any actual or prospective purchaser or owner of this vehicle .
All dealer fees and amounts must be disclosed in a clear, conspicuous and legible fashion. There are no exclusions .
```

**public_25** — 4.57 seconds; 318 input tokens; source chunks 1

```text
The listed vehicle is 2015 fiat 500l with 57,244 miles .
The listed condition is good; fuel: gas; transmission: manual; drivetrain: fwd; body type: SUV .
Vehicle description (unverified):
6 Speed Manual Folding rear seats USB and Auxiliary connections Cruise Control No dealer fee corolla yaris accent prius c v civic fit insight avalon leaf volt bolt plugin plug-in sentra versa camry Altima accord focus cobalt fiesta cruze spark speed coupe hatchback town car elantra azure genesis 128i 135i 325i 328i 330i 525
```

**public_50** — 2.31 seconds; 870 input tokens; source chunks 1

```text
The listed vehicle is 2011 hyundai equus signature with 74,900 miles .
The car is fully loaded with everything offered in 2011 and has more features that you can imagine .
Low mileage at 73,900 .
```

**public_75** — 1.79 seconds; 576 input tokens; source chunks 1

```text
The listed vehicle is 2018 lexus es with 26,485 miles .
The listed car is 2018 Lexus ES ES ES 350 .
Vehicle has been registered in the US .
```


## Reproduction and attribution

`python scripts/test_bart_summary.py` uses cases from `scripts/test_smollm_summary.py`.
Python 3.12.14; PyTorch 2.15.0+cpu; Transformers 5.21.0.
Publisher supplies a PyTorch state dict; loading uses weights_only=True and disables remote code.

Four public examples: Tom, Used Cars Dataset, 2021 asking listings,
https://zenodo.org/records/10457828 ; CC BY 4.0 https://creativecommons.org/licenses/by/4.0/.
Existing examples redact prices and common contacts/VINs. Full inputs and outputs
are in results.json. Model card: https://huggingface.co/ainize/bart-base-cnn .
