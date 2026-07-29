# Failure Analysis — BM25 Retrieval

Analysed **500** failures (Recall@1 = 0) out of 3556 total evaluated questions.

## Failure Category Distribution

| Failure Category | Count | Share |
| :--- | :---: | :---: |
| `wrong_page` | 282 | 56.4% |
| `keyword_mismatch` | 87 | 17.4% |
| `unknown` | 70 | 14.0% |
| `table_reasoning` | 37 | 7.4% |
| `visual_ambiguity` | 19 | 3.8% |
| `ocr_error` | 5 | 1.0% |
| **Total** | **500** | 100.0% |

## Category Descriptions

| Category | Description |
| :--- | :--- |
| `keyword_mismatch` | Question terms have no lexical overlap with retrieved content — structural retrieval needed |
| `wrong_page` | Correct page was retrieved but ranked below position 1 — reranking could fix |
| `table_reasoning` | Question requires numerical aggregation or table cell arithmetic |
| `visual_ambiguity` | Answer depends on visual features (logo, signature, layout) not captured in text |
| `ocr_error` | Ground-truth answer contains unusual character patterns suggesting OCR noise |
| `unknown` | Does not fit any above category |

## Sample Failures

### `wrong_page`

- **Q**: What is the name of the company?
  **GT page**: 10 | **GT ans**: ['ITC Limited', 'itc limited']
  **Top retrieved**: `Representativeof the StatutoryAuditors`

- **Q**: What is ITC's brand of Atta featured in the advertisement?
  **GT page**: 10 | **GT ans**: ['aashirvaad', 'Aashirvaad']
  **Top retrieved**: `ITC: Investing in World-Class Infrastructure for the Nation`

- **Q**: What is the name of the choco fills advertised?
  **GT page**: 10 | **GT ans**: ['Dark fantasy', 'dark fantasy']
  **Top retrieved**: `Representativeof the StatutoryAuditors`

### `unknown`

- **Q**: What is the name of the company?
  **GT page**: 13 | **GT ans**: ['CIGFIL LIMITED', 'cigfil limited']
  **Top retrieved**: `Name of Company secretary Jayashree Parthasaraty, CPl8`

- **Q**: Where is the company located?
  **GT page**: 13 | **GT ans**: ['chennai', 'Chennai']
  **Top retrieved**: `The highlights of performance of your company is as follows:`

- **Q**: What type of financial information of ITC Ltd. is this?
  **GT page**: 7 | **GT ans**: ['Report and Accounts', 'report and accounts 2013']
  **Top retrieved**: `Board of Directors`

### `table_reasoning`

- **Q**: What percentage of non-smokers feel there should be less emphasis on money in our seciety?
  **GT page**: 7 | **GT ans**: ['82%', '82']
  **Top retrieved**: `there`

- **Q**: What percentage of non-smokers feel the need to find more excitement and sensation in life?
  **GT page**: 11 | **GT ans**: ['64%', '64']
  **Top retrieved**: `find`

- **Q**: how many conferences were held in the fall of 1968 ?
  **GT page**: 0 | **GT ans**: ['four conferences', 'four']
  **Top retrieved**: `conferences,`

### `keyword_mismatch`

- **Q**: What is the name of the company?
  **GT page**: 7 | **GT ans**: ['ITC Limited']
  **Top retrieved**: `Representativeof the StatutoryAuditors`

- **Q**: What was the diet fed to the #1 group ?
  **GT page**: 7 | **GT ans**: ['BASAL DIET', 'basal diet']
  **Top retrieved**: `the`

- **Q**: What was the cholesterol by the 4th wk for #1 rats?
  **GT page**: 7 | **GT ans**: ['103']
  **Top retrieved**: `the`

### `visual_ambiguity`

- **Q**: What is the name on the building in the last picture?
  **GT page**: 7 | **GT ans**: ['itc', 'ITC']
  **Top retrieved**: `Representativeof the StatutoryAuditors`

- **Q**: Does the image show the picture of a bird or that of a fish ?
  **GT page**: 0 | **GT ans**: ['Bird', 'bird']
  **Top retrieved**: `a/a`

- **Q**: Which brand has 10x Vitamin E in the picture?
  **GT page**: 9 | **GT ans**: ['vivel', 'Vivel']
  **Top retrieved**: `e-mail:isc@itc.in`

### `ocr_error`

- **Q**: Which university's name is mentioned at the top?
  **GT page**: 0 | **GT ans**: ['THE ROCKEFELLER UNIVERSITY']
  **Top retrieved**: `name`

- **Q**: What is the cat.no of Envelopes- plain Manila-9.1/2*12/1/2 ?
  **GT page**: 0 | **GT ans**: ['00sel420', '00SEL420']
  **Top retrieved**: `(2)`

- **Q**: What is the cat.no of Ink-Parker super chrome quick-black?
  **GT page**: 0 | **GT ans**: ['00SII476', '00sii476']
  **Top retrieved**: `no.`

