# Reproduction of Mamba1 and Mamba2

> Reproduced by AkagawaTsurunaki@Github

Origin Github repo: [state-spaces/mamba: Mamba SSM architecture](https://github.com/state-spaces/mamba)

## Paper Info

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**

Albert Gu\*, Tri Dao\*

Paper: https://arxiv.org/abs/2312.00752

**Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**

Tri Dao\*, Albert Gu\*

Paper: https://arxiv.org/abs/2405.21060

## Results

> [!NOTE]
>
> 1. Some experiments or values are not found in the origin paper but are shown for completeness.
> 2. We may use less `batch_size` due to the limited GPU resource. 
> 3. **†Value** means the data is copied from origin paper, while **Value** is the reproduced data by AkagawaTsurunaki.

### Mamba 1

See Zero-shot Evaluations, Table 3 of *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 

> We report accuracy for LAMBADA, WinoGrande, PIQA, and ARC-easy, and accuracy normalized by sequence length for HellaSwag and ARC-challenge (since normalized accuracy is higher for almost all models for these task).

**mamba_ssm (pretrained=state-spaces/mamba-130m), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256**

| Tasks          | Metric     |      | †Value |   Value |      | Stderr |
| -------------- | ---------- | ---- | ------ | ------: | ---- | -----: |
| arc_challenge  | acc        | ↑    |        |  0.1971 | ±    | 0.0116 |
|                | acc_norm   | ↑    | 24.3   |  0.2406 | ±    | 0.0125 |
| arc_easy       | acc        | ↑    | 48.0   |  0.4798 | ±    | 0.0103 |
|                | acc_norm   | ↑    |        |  0.4205 | ±    | 0.0101 |
| hellaswag      | acc        | ↑    |        |  0.3077 | ±    | 0.0046 |
|                | acc_norm   | ↑    | 35.3   |  0.3525 | ±    | 0.0048 |
| lambada_openai | acc        | ↑    | 44.3   |  0.4417 | ±    | 0.0069 |
|                | perplexity | ↓    | 16.07  | 16.0450 | ±    | 0.5092 |
| openbookqa     | acc        | ↑    |        |  0.1660 | ±    | 0.0167 |
|                | acc_norm   | ↑    |        |  0.2840 | ±    | 0.0202 |
| piqa           | acc        | ↑    | 64.5   |  0.6453 | ±    | 0.0112 |
|                | acc_norm   | ↑    |        |  0.6333 | ±    | 0.0112 |
| winogrande     | acc        | ↑    | 51.9   |  0.5249 | ±    | 0.0140 |

**hf (pretrained=EleutherAI/pythia-160m), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64**

| Tasks          | Metric     |      | †Value |   Value |      | Stderr |
| -------------- | ---------- | ---- | ------ | ------: | ---- | -----: |
| arc_challenge  | acc        | ↑    |        |  0.1877 | ±    | 0.0114 |
|                | acc_norm   | ↑    | 24.1   |  0.2406 | ±    | 0.0125 |
| arc_easy       | acc        | ↑    | 43.2   |  0.4364 | ±    | 0.0102 |
|                | acc_norm   | ↑    |        |  0.3935 | ±    | 0.0100 |
| hellaswag      | acc        | ↑    |        |  0.2842 | ±    | 0.0045 |
|                | acc_norm   | ↑    | 30.2   |  0.3028 | ±    | 0.0046 |
| lambada_openai | acc        | ↑    | 33.0   |  0.3264 | ±    | 0.0065 |
|                | perplexity | ↓    | 38.10  | 38.0065 | ±    | 1.4326 |
| piqa           | acc        | ↑    |        |  0.6148 | ±    | 0.0114 |
|                | acc_norm   | ↑    | 61.4   |  0.6175 | ±    | 0.0113 |
| winogrande     | acc        | ↑    | 51.9   |  0.5178 | ±    | 0.0140 |

**mamba_ssm (pretrained=state-spaces/mamba-370m), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256**

| Tasks          | Metric     |      | †Value |  Value |      | Stderr |
| -------------- | ---------- | ---- | ------ | -----: | ---- | -----: |
| arc_challenge  | acc        | ↑    |        | 0.2474 | ±    | 0.0126 |
|                | acc_norm   | ↑    | 28.0   | 0.2799 | ±    | 0.0131 |
| arc_easy       | acc        | ↑    | 55.1   | 0.5492 | ±    | 0.0102 |
|                | acc_norm   | ↑    |        | 0.4819 | ±    | 0.0103 |
| hellaswag      | acc        | ↑    |        | 0.3720 | ±    | 0.0048 |
|                | acc_norm   | ↑    | 46.5   | 0.4646 | ±    | 0.0050 |
| lambada_openai | acc        | ↑    | 55.6   | 0.5560 | ±    | 0.0069 |
|                | perplexity | ↓    | 8.14   | 8.1375 | ±    | 0.2232 |
| openbookqa     | acc        | ↑    |        | 0.1960 | ±    | 0.0178 |
|                | acc_norm   | ↑    |        | 0.3120 | ±    | 0.0207 |
| piqa           | acc        | ↑    | 69.5   | 0.6942 | ±    | 0.0107 |
|                | acc_norm   | ↑    |        | 0.6844 | ±    | 0.0108 |
| winogrande     | acc        | ↑    | 55.3   | 0.5556 | ±    | 0.0140 |

**mamba_ssm (pretrained=state-spaces/mamba-790M), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256** 

| Tasks          | Metric     |      | †Value |  Value |      | Stderr |
| -------------- | ---------- | ---- | ------ | -----: | ---- | -----: |
| arc_challenge  | acc        | ↑    |        | 0.2637 | ±    | 0.0129 |
|                | acc_norm   | ↑    | 29.5   | 0.2935 | ±    | 0.0133 |
| arc_easy       | acc        | ↑    | 61.2   | 0.6107 | ±    | 0.0100 |
|                | acc_norm   | ↑    |        | 0.5391 | ±    | 0.0102 |
| hellaswag      | acc        | ↑    |        | 0.4230 | ±    | 0.0049 |
|                | acc_norm   | ↑    | 55.1   | 0.5501 | ±    | 0.0050 |
| lambada_openai | acc        | ↑    | 62.7   | 0.6146 | ±    | 0.0068 |
|                | perplexity | ↓    | 6.02   | 6.0149 | ±    | 0.1513 |
| openbookqa     | acc        | ↑    |        | 0.2300 | ±    | 0.0188 |
|                | acc_norm   | ↑    |        | 0.3420 | ±    | 0.0212 |
| piqa           | acc        | ↑    | 72.1   | 0.7193 | ±    | 0.0105 |
|                | acc_norm   | ↑    |        | 0.7247 | ±    | 0.0104 |
| winogrande     | acc        | ↑    | 56.1   | 0.5627 | ±    | 0.0139 |

**mamba_ssm (pretrained=state-spaces/mamba-1.4B), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256**

| Tasks          | Metric     |      | †Value |  Value |      | Stderr |
| -------------- | ---------- | ---- | ------ | -----: | ---- | -----: |
| arc_challenge  | acc        | ↑    |        | 0.2978 | ±    | 0.0134 |
|                | acc_norm   | ↑    | 32.8   | 0.3268 | ±    | 0.0137 |
| arc_easy       | acc        | ↑    | 65.5   | 0.6553 | ±    | 0.0098 |
|                | acc_norm   | ↑    |        | 0.6120 | ±    | 0.0100 |
| hellaswag      | acc        | ↑    |        | 0.4502 | ±    | 0.0050 |
|                | acc_norm   | ↑    | 59.1   | 0.5909 | ±    | 0.0049 |
| lambada_openai | acc        | ↑    | 64.9   | 0.6491 | ±    | 0.0066 |
|                | perplexity | ↓    | 5.04   | 5.0423 | ±    | 0.1205 |
| openbookqa     | acc        | ↑    |        | 0.2620 | ±    | 0.0197 |
|                | acc_norm   | ↑    |        | 0.3640 | ±    | 0.0215 |
| piqa           | acc        | ↑    | 74.2   | 0.7405 | ±    | 0.0102 |
|                | acc_norm   | ↑    |        | 0.7399 | ±    | 0.0102 |
| winogrande     | acc        | ↑    | 61.5   | 0.6109 | ±    | 0.0137 |

**mamba_ssm (pretrained=state-spaces/mamba-2.8B), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256**

| Tasks          | Metric     |      | †Value |  Value |      | Stderr |
| -------------- | ---------- | ---- | ------ | -----: | ---- | -----: |
| arc_challenge  | acc        | ↑    |        | 0.3439 | ±    | 0.0139 |
|                | acc_norm   | ↑    | 36.3   | 0.3635 | ±    | 0.0141 |
| arc_easy       | acc        | ↑    | 69.7   | 0.6961 | ±    | 0.0094 |
|                | acc_norm   | ↑    |        | 0.6427 | ±    | 0.0098 |
| hellaswag      | acc        | ↑    |        | 0.4951 | ±    | 0.0050 |
|                | acc_norm   | ↑    | 66.1   | 0.6612 | ±    | 0.0047 |
| lambada_openai | acc        | ↑    | 69.2   | 0.6909 | ±    | 0.0064 |
|                | perplexity | ↓    | 4.23   | 4.2303 | ±    | 0.0949 |
| openbookqa     | acc        | ↑    |        | 0.2940 | ±    | 0.0204 |
|                | acc_norm   | ↑    |        | 0.3960 | ±    | 0.0219 |
| piqa           | acc        | ↑    | 75.2   | 0.7530 | ±    | 0.0101 |
|                | acc_norm   | ↑    |        | 0.7584 | ±    | 0.0100 |
| winogrande     | acc        | ↑    | 63.5   | 0.6338 | ±    | 0.0135 |

### Mamba-2

#### Report blog?

> [!NOTE]
>
> I do not find this blog... Let me know if you have any idea.

**mamba_ssm (pretrained=state-spaces/mamba-2.8b-slimpj), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32**

|    Tasks     |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge |      1|none  |     0|acc     |↑  |0.3857|±  |0.0142|
|              |       |none  |     0|acc_norm|↑  |0.4164|±  |0.0144|
|arc_easy      |      1|none  |     0|acc     |↑  |0.7260|±  |0.0092|
|              |       |none  |     0|acc_norm|↑  |0.6818|±  |0.0096|
|boolq         |      2|none  |     0|acc     |↑  |0.7104|±  |0.0079|
|hellaswag     |      1|none  |     0|acc     |↑  |0.5248|±  |0.0050|
|              |       |none  |     0|acc_norm|↑  |0.7097|±  |0.0045|
|openbookqa    |      1|none  |     0|acc     |↑  |0.2860|±  |0.0202|
|              |       |none  |     0|acc_norm|↑  |0.3980|±  |0.0219|
|piqa          |      1|none  |     0|acc     |↑  |0.7709|±  |0.0098|
|              |       |none  |     0|acc_norm|↑  |0.7818|±  |0.0096|
|race          |      2|none  |     0|acc     |↑  |0.3675|±  |0.0149|
|truthfulqa_mc2|      3|none  |     0|acc     |↑  |0.3436|±  |0.0132|
|winogrande    |      1|none  |     0|acc     |↑  |0.6598|±  |0.0133|

**mba_ssm (pretrained=state-spaces/mamba-2.8b-slimpj), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 16**

| Tasks                                 | Version | n-shot | Metric |      |  Value |      | Stderr |
| ------------------------------------- | ------: | -----: | ------ | ---- | -----: | ---- | -----: |
| mmlu                                  |       2 |        | acc    | ↑    | 0.2613 | ±    | 0.0037 |
| - humanities                          |       2 |        | acc    | ↑    | 0.2506 | ±    | 0.0063 |
| - formal_logic                        |       1 |      5 | acc    | ↑    | 0.1825 | ±    | 0.0346 |
| - high_school_european_history        |       1 |      5 | acc    | ↑    | 0.2606 | ±    | 0.0343 |
| - high_school_us_history              |       1 |      5 | acc    | ↑    | 0.2206 | ±    | 0.0291 |
| - high_school_world_history           |       1 |      5 | acc    | ↑    | 0.2616 | ±    | 0.0286 |
| - international_law                   |       1 |      5 | acc    | ↑    | 0.2645 | ±    | 0.0403 |
| - jurisprudence                       |       1 |      5 | acc    | ↑    | 0.2593 | ±    | 0.0424 |
| - logical_fallacies                   |       1 |      5 | acc    | ↑    | 0.2515 | ±    | 0.0341 |
| - moral_disputes                      |       1 |      5 | acc    | ↑    | 0.2399 | ±    | 0.0230 |
| - moral_scenarios                     |       1 |      5 | acc    | ↑    | 0.2503 | ±    | 0.0145 |
| - philosophy                          |       1 |      5 | acc    | ↑    | 0.2733 | ±    | 0.0253 |
| - prehistory                          |       1 |      5 | acc    | ↑    | 0.2870 | ±    | 0.0252 |
| - professional_law                    |       1 |      5 | acc    | ↑    | 0.2425 | ±    | 0.0109 |
| - world_religions                     |       1 |      5 | acc    | ↑    | 0.2807 | ±    | 0.0345 |
| - other                               |       2 |        | acc    | ↑    | 0.2713 | ±    | 0.0080 |
| - business_ethics                     |       1 |      5 | acc    | ↑    | 0.2500 | ±    | 0.0435 |
| - clinical_knowledge                  |       1 |      5 | acc    | ↑    | 0.2830 | ±    | 0.0277 |
| - college_medicine                    |       1 |      5 | acc    | ↑    | 0.2312 | ±    | 0.0321 |
| - global_facts                        |       1 |      5 | acc    | ↑    | 0.3100 | ±    | 0.0465 |
| - human_aging                         |       1 |      5 | acc    | ↑    | 0.3677 | ±    | 0.0324 |
| - management                          |       1 |      5 | acc    | ↑    | 0.2427 | ±    | 0.0425 |
| - marketing                           |       1 |      5 | acc    | ↑    | 0.2479 | ±    | 0.0283 |
| - medical_genetics                    |       1 |      5 | acc    | ↑    | 0.2600 | ±    | 0.0441 |
| - miscellaneous                       |       1 |      5 | acc    | ↑    | 0.2733 | ±    | 0.0159 |
| - nutrition                           |       1 |      5 | acc    | ↑    | 0.2353 | ±    | 0.0243 |
| - professional_accounting             |       1 |      5 | acc    | ↑    | 0.2589 | ±    | 0.0261 |
| - professional_medicine               |       1 |      5 | acc    | ↑    | 0.2390 | ±    | 0.0259 |
| - virology                            |       1 |      5 | acc    | ↑    | 0.3434 | ±    | 0.0370 |
| - social sciences                     |       2 |        | acc    | ↑    | 0.2554 | ±    | 0.0079 |
| - econometrics                        |       1 |      5 | acc    | ↑    | 0.3246 | ±    | 0.0440 |
| - high_school_geography               |       1 |      5 | acc    | ↑    | 0.2374 | ±    | 0.0303 |
| - high_school_government_and_politics |       1 |      5 | acc    | ↑    | 0.2332 | ±    | 0.0305 |
| - high_school_macroeconomics          |       1 |      5 | acc    | ↑    | 0.2718 | ±    | 0.0226 |
| - high_school_microeconomics          |       1 |      5 | acc    | ↑    | 0.2563 | ±    | 0.0284 |
| - high_school_psychology              |       1 |      5 | acc    | ↑    | 0.2459 | ±    | 0.0185 |
| - human_sexuality                     |       1 |      5 | acc    | ↑    | 0.2366 | ±    | 0.0373 |
| - professional_psychology             |       1 |      5 | acc    | ↑    | 0.2500 | ±    | 0.0175 |
| - public_relations                    |       1 |      5 | acc    | ↑    | 0.3455 | ±    | 0.0455 |
| - security_studies                    |       1 |      5 | acc    | ↑    | 0.2449 | ±    | 0.0275 |
| - sociology                           |       1 |      5 | acc    | ↑    | 0.2388 | ±    | 0.0301 |
| - us_foreign_policy                   |       1 |      5 | acc    | ↑    | 0.2600 | ±    | 0.0441 |
| - stem                                |       2 |        | acc    | ↑    | 0.2731 | ±    | 0.0079 |
| - abstract_algebra                    |       1 |      5 | acc    | ↑    | 0.2700 | ±    | 0.0446 |
| - anatomy                             |       1 |      5 | acc    | ↑    | 0.2000 | ±    | 0.0346 |
| - astronomy                           |       1 |      5 | acc    | ↑    | 0.1908 | ±    | 0.0320 |
| - college_biology                     |       1 |      5 | acc    | ↑    | 0.2569 | ±    | 0.0365 |
| - college_chemistry                   |       1 |      5 | acc    | ↑    | 0.2800 | ±    | 0.0451 |
| - college_computer_science            |       1 |      5 | acc    | ↑    | 0.2700 | ±    | 0.0446 |
| - college_mathematics                 |       1 |      5 | acc    | ↑    | 0.2700 | ±    | 0.0446 |
| - college_physics                     |       1 |      5 | acc    | ↑    | 0.1961 | ±    | 0.0395 |
| - computer_security                   |       1 |      5 | acc    | ↑    | 0.2600 | ±    | 0.0441 |
| - conceptual_physics                  |       1 |      5 | acc    | ↑    | 0.2638 | ±    | 0.0288 |
| - electrical_engineering              |       1 |      5 | acc    | ↑    | 0.2483 | ±    | 0.0360 |
| - elementary_mathematics              |       1 |      5 | acc    | ↑    | 0.2566 | ±    | 0.0225 |
| - high_school_biology                 |       1 |      5 | acc    | ↑    | 0.2710 | ±    | 0.0253 |
| - high_school_chemistry               |       1 |      5 | acc    | ↑    | 0.2709 | ±    | 0.0313 |
| - high_school_computer_science        |       1 |      5 | acc    | ↑    | 0.2900 | ±    | 0.0456 |
| - high_school_mathematics             |       1 |      5 | acc    | ↑    | 0.2667 | ±    | 0.0270 |
| - high_school_physics                 |       1 |      5 | acc    | ↑    | 0.3377 | ±    | 0.0386 |
| - high_school_statistics              |       1 |      5 | acc    | ↑    | 0.4074 | ±    | 0.0335 |
| - machine_learning                    |       1 |      5 | acc    | ↑    | 0.3482 | ±    | 0.0452 |

| Groups            | Version | Metric |      |  Value |      | Stderr |
| ----------------- | ------: | ------ | ---- | -----: | ---- | -----: |
| mmlu              |       2 | acc    | ↑    | 0.2613 | ±    | 0.0037 |
| - humanities      |       2 | acc    | ↑    | 0.2506 | ±    | 0.0063 |
| - other           |       2 | acc    | ↑    | 0.2713 | ±    | 0.0080 |
| - social sciences |       2 | acc    | ↑    | 0.2554 | ±    | 0.0079 |
| - stem            |       2 | acc    | ↑    | 0.2731 | ±    | 0.0079 |

---

#### Zero-shot Evaluations

See Zero-shot Evaluations, Table3 of *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*.

**mamba_ssm (pretrained=state-spaces/mamba2-2.7b), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256**

| Tasks          | Version | Filter | n-shot | Metric     |      | †Value |  Value |      | Stderr |
| -------------- | ------: | ------ | -----: | ---------- | ---- | ------ | -----: | ---- | -----: |
| arc_challenge  |       1 | none   |      0 | acc        | ↑    |        | 0.3319 | ±    | 0.0138 |
|                |         | none   |      0 | acc_norm   | ↑    | 36.4   | 0.3643 | ±    | 0.0141 |
| arc_easy       |       1 | none   |      0 | acc        | ↑    | 69.6   | 0.6957 | ±    | 0.0094 |
|                |         | none   |      0 | acc_norm   | ↑    |        | 0.6481 | ±    | 0.0098 |
| hellaswag      |       1 | none   |      0 | acc        | ↑    |        | 0.4962 | ±    | 0.0050 |
|                |         | none   |      0 | acc_norm   | ↑    | 66.6   | 0.6654 | ±    | 0.0047 |
| lambada_openai |       1 | none   |      0 | acc        | ↑    | 69.7   | 0.6947 | ±    | 0.0064 |
|                |         | none   |      0 | perplexity | ↓    | 4.10   | 4.0932 | ±    | 0.0890 |
| openbookqa     |       1 | none   |      0 | acc        | ↑    |        | 0.2920 | ±    | 0.0204 |
|                |         | none   |      0 | acc_norm   | ↑    | 38.8   | 0.3880 | ±    | 0.0218 |
| piqa           |       1 | none   |      0 | acc        | ↑    | 76.4   | 0.7633 | ±    | 0.0099 |
|                |         | none   |      0 | acc_norm   | ↑    |        | 0.7617 | ±    | 0.0099 |
| winogrande     |       1 | none   |      0 | acc        | ↑    | 64.0   | 0.6393 | ±    | 0.0135 |

**mamba_ssm (pretrained=state-spaces/transformerpp-2.7b), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256**

| Tasks          | Version | Filter | n-shot | Metric     |      | †Value |  Value |      | Stderr |
| -------------- | ------: | ------ | -----: | ---------- | ---- | ------ | -----: | ---- | -----: |
| arc_challenge  |       1 | none   |      0 | acc        | ↑    |        | 0.3242 | ±    | 0.0137 |
|                |         | none   |      0 | acc_norm   | ↑    | 37.8   | 0.3780 | ±    | 0.0142 |
| arc_easy       |       1 | none   |      0 | acc        | ↑    | 67.7   | 0.6776 | ±    | 0.0096 |
|                |         | none   |      0 | acc_norm   | ↑    |        | 0.6305 | ±    | 0.0099 |
| hellaswag      |       1 | none   |      0 | acc        | ↑    |        | 0.4971 | ±    | 0.0050 |
|                |         | none   |      0 | acc_norm   | ↑    | 66.4   | 0.6637 | ±    | 0.0047 |
| lambada_openai |       1 | none   |      0 | acc        | ↑    | 70.3   | 0.7027 | ±    | 0.0064 |
|                |         | none   |      0 | perplexity | ↓    | 3.99   | 3.9878 | ±    | 0.0860 |
| openbookqa     |       1 | none   |      0 | acc        | ↑    |        | 0.2900 | ±    | 0.0203 |
|                |         | none   |      0 | acc_norm   | ↑    | 40.4   | 0.4060 | ±    | 0.0220 |
| piqa           |       1 | none   |      0 | acc        | ↑    | 75.2   | 0.7519 | ±    | 0.0101 |
|                |         | none   |      0 | acc_norm   | ↑    |        | 0.7644 | ±    | 0.0099 |
| winogrande     |       1 | none   |      0 | acc        | ↑    | 63.9   | 0.6377 | ±    | 0.0135 |

**mamba_ssm (pretrained=state-spaces/mamba2attn-2.7b), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64**

| Tasks          | Version | Filter | n-shot | Metric     |      | †Value |  Value |      | Stderr |
| -------------- | ------: | ------ | -----: | ---------- | ---- | ------ | -----: | ---- | -----: |
| arc_challenge  |       1 | none   |      0 | acc        | ↑    |        | 0.3456 | ±    | 0.0139 |
|                |         | none   |      0 | acc_norm   | ↑    | 37.8   | 0.3788 | ±    | 0.0142 |
| arc_easy       |       1 | none   |      0 | acc        | ↑    | 69.9   | 0.6987 | ±    | 0.0094 |
|                |         | none   |      0 | acc_norm   | ↑    |        | 0.6629 | ±    | 0.0097 |
| hellaswag      |       1 | none   |      0 | acc        | ↑    |        | 0.5025 | ±    | 0.0050 |
|                |         | none   |      0 | acc_norm   | ↑    | 67.8   | 0.6779 | ±    | 0.0047 |
| lambada_openai |       1 | none   |      0 | acc        | ↑    | 71.1   | 0.7105 | ±    | 0.0063 |
|                |         | none   |      0 | perplexity | ↓    | 3.85   | 3.8489 | ±    | 0.0810 |
| openbookqa     |       1 | none   |      0 | acc        | ↑    |        | 0.3040 | ±    | 0.0206 |
|                |         | none   |      0 | acc_norm   | ↑    | 39.0   | 0.3900 | ±    | 0.0218 |
| piqa           |       1 | none   |      0 | acc        | ↑    | 75.8   | 0.7573 | ±    | 0.0100 |
|                |         | none   |      0 | acc_norm   | ↑    |        | 0.7595 | ±    | 0.0100 |
| winogrande     |       1 | none   |      0 | acc        | ↑    | 65.3   | 0.6511 | ±    | 0.0134 |

## Reproduction Details

All results were reproduced using **Ubuntu 22.04** with **CUDA 11.8** and were evaluated on the official models provided on Hugging Face. 

### Hardware Info

- CPU: Intel i9 14900HF
- GPU: NVIDIA GeForce RTX 4090 (Driver Version: 580.82.07)
- Memory: 192 GB RAM

### Environment of Anaconda

**Python 3.10.19**

You should install these 3 special packages manually:

- `causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`: [Release v1.4.0 · Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.4.0)
- `mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`: [Release v2.2.2 · state-spaces/mamba](https://github.com/state-spaces/mamba/releases/tag/v2.2.2)
- `flash_attn-2.7.3+cu11torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`: [Releases · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/releases)

This is due to ABI incompatibility, which can cause undefined symbol errors at runtime.

The following code block is the `environment.yml` created by Anaconda: 

``` yaml
name: mamba
channels:
  - nvidia/label/cuda-11.8.0
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - bzip2=1.0.8=h5eee18b_6
  - ca-certificates=2025.9.9=h06a4308_0
  - cuda-nvcc=11.8.89=0
  - cudatoolkit=11.8.0=h6a678d5_0
  - expat=2.7.1=h6a678d5_0
  - ld_impl_linux-64=2.44=h153f514_2
  - libffi=3.4.4=h6a678d5_1
  - libgcc-ng=11.2.0=h1234567_1
  - libgomp=11.2.0=h1234567_1
  - libnsl=2.0.0=h5eee18b_0
  - libstdcxx-ng=11.2.0=h1234567_1
  - libuuid=1.41.5=h5eee18b_0
  - libxcb=1.17.0=h9b100fa_0
  - libzlib=1.3.1=hb25bd0a_0
  - ncurses=6.5=h7934f7d_0
  - openssl=3.0.18=hd6dcaed_0
  - pip=25.2=pyhc872135_1
  - pthread-stubs=0.3=h0ce48e5_1
  - python=3.10.19=h6fa692b_0
  - readline=8.3=hc2a1206_0
  - setuptools=80.9.0=py310h06a4308_0
  - sqlite=3.50.2=hb25bd0a_1
  - tk=8.6.15=h54e0aa7_0
  - wheel=0.45.1=py310h06a4308_0
  - xorg-libx11=1.8.12=h9b100fa_1
  - xorg-libxau=1.0.12=h9b100fa_0
  - xorg-libxdmcp=1.1.5=h9b100fa_0
  - xorg-xorgproto=2024.1=h5eee18b_1
  - xz=5.6.4=h5eee18b_1
  - zlib=1.3.1=hb25bd0a_0
  - pip:
      - absl-py==2.3.1
      - accelerate==1.11.0
      - aiohappyeyeballs==2.6.1
      - aiohttp==3.13.2
      - aiosignal==1.4.0
      - anyio==4.11.0
      - async-timeout==5.0.1
      - attrs==25.4.0
      - causal-conv1d==1.4.0
      - certifi==2025.10.5
      - chardet==5.2.0
      - charset-normalizer==3.4.4
      - click==8.3.0
      - colorama==0.4.6
      - dataproperty==1.1.0
      - datasets==3.6.0
      - dill==0.3.8
      - einops==0.8.1
      - evaluate==0.4.6
      - exceptiongroup==1.3.0
      - filelock==3.19.1
      - flash-attn==2.7.3
      - frozenlist==1.8.0
      - fsspec==2025.3.0
      - h11==0.16.0
      - hf-xet==1.2.0
      - httpcore==1.0.9
      - httpx==0.28.1
      - huggingface-hub==0.36.0
      - idna==3.11
      - jinja2==3.1.6
      - joblib==1.5.2
      - jsonlines==4.0.0
      - lm-eval==0.4.9.1
      - lxml==6.0.2
      - mamba-ssm==2.2.2
      - markupsafe==2.1.5
      - mbstrdecoder==1.1.4
      - more-itertools==10.8.0
      - mpmath==1.3.0
      - multidict==6.7.0
      - multiprocess==0.70.16
      - networkx==3.3
      - ninja==1.13.0
      - nltk==3.9.2
      - numexpr==2.14.1
      - numpy==2.1.2
      - nvidia-cublas-cu11==11.11.3.6
      - nvidia-cuda-cupti-cu11==11.8.87
      - nvidia-cuda-nvrtc-cu11==11.8.89
      - nvidia-cuda-runtime-cu11==11.8.89
      - nvidia-cudnn-cu11==8.7.0.84
      - nvidia-cufft-cu11==10.9.0.58
      - nvidia-curand-cu11==10.3.0.86
      - nvidia-cusolver-cu11==11.4.1.48
      - nvidia-cusparse-cu11==11.7.5.86
      - nvidia-nccl-cu11==2.20.5
      - nvidia-nvtx-cu11==11.8.86
      - packaging==25.0
      - pandas==2.3.3
      - pathvalidate==3.3.1
      - peft==0.10.0
      - pillow==11.3.0
      - portalocker==3.2.0
      - propcache==0.4.1
      - psutil==7.1.3
      - pyarrow==22.0.0
      - pybind11==3.0.1
      - pytablewriter==1.2.1
      - python-dateutil==2.9.0.post0
      - pytz==2025.2
      - pyyaml==6.0.3
      - regex==2025.10.23
      - requests==2.32.5
      - rouge-score==0.1.2
      - sacrebleu==2.5.1
      - safetensors==0.6.2
      - scikit-learn==1.7.2
      - scipy==1.15.3
      - six==1.17.0
      - sniffio==1.3.1
      - sqlitedict==2.1.0
      - sympy==1.14.0
      - tabledata==1.3.4
      - tabulate==0.9.0
      - tcolorpy==0.1.7
      - threadpoolctl==3.6.0
      - tokenizers==0.22.1
      - torch==2.3.1+cu118
      - torchaudio==2.3.1+cu118
      - torchvision==0.18.1+cu118
      - tqdm==4.67.1
      - tqdm-multiprocess==0.0.11
      - transformers==4.57.1
      - triton==2.3.1
      - typepy==1.3.4
      - typing-extensions==4.15.0
      - tzdata==2025.2
      - urllib3==2.5.0
      - word2number==1.1
      - xxhash==3.6.0
      - yarl==1.22.0
      - zstandard==0.25.0
```

If you have any questions, contact with me through AkagawaTsurunaki@outlook.com. 

