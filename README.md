# Medical Abstraction and Reasoning Corpus (MedARC-QA)

[**📖 Paper**]() |

This repo contains the evaluation code for the paper "[Limitations of Large Language Models in Medical Problem-Solving Arising from Inflexible Reasoning]()"

## Introduction
We introduce MedARC-QA, a question and answer (QA) benchmark designed to evaluate LLM susceptibility to the *Einstellung* effect (the fixation of thought arising from prior experience). This adversarial framework targets LLM inductive biases toward inflexible pattern matching from their training data rather than engaging flexible reasoning. Our results show that a LLMs show poor performance on MedARC-QA contrasting with findings on conventional medical QA (e.g. USMLE). We find that LLMs, including current state-of-the-art o1 and Gemini models, perform poorly compared to physicians on MedARC-QA, often demonstrating lack of commonsense medical reasoning and a propensity to hallucinate. In addition, uncertainty estimation analyses indicate that LLMs exhibit overconfidence in their answers, despite their limited accuracy. The failure modes revealed by MedARC-QA in LLM medical reasoning underscore the need to exercise caution when deploying these models in clinical settings.

The `MedARC-QA` dataset is located in the `data_medARC_v1` folder.

## Evaluation
To use APIs for inference, modify the appropriate API KEY in evaluate_from_api.py script and execute the corresponding bash script:

```bash
cd v1_scripts/
sh eval_gpt_4o.sh
```

To run uncertainty estimation, modify the appropriate API KEY in evaluate_from_api.py script and execute the corresponding bash script:

```bash
cd v1_UQ_scripts/
sh eval_uq_gpt_4o.sh
```

## Results
Jupyter notebooks, Results_v1(20250126).ipynb and Results_UQ_v1(20250126).ipynb, reproduce Figure 2 and Figure 6 from the manuscript. 
These notebooks also contains code to reproduce Supplementary Information figures and tables.

## 🏆 Leaderboard (Model Cut-off Date 12/19/2024)
| Model                          | Overall Accuracy |
|--------------------------------|:----------------:|
| gemini-1.5-pro-latest          | 50.00           |
| o1-2024-12-17_200000           | 48.00           |
| Llama-3.1-70B                  | 44.00           |
| claude-3-opus                  | 38.00           |
| Llama3.1-405b-instruct-fp8     | 34.00           |
| Llama-3.3-70B                  | 34.00           |
| o1-mini                        | 31.00           |
| claude-3-sonnet                | 29.00           |
| gemini-1.5-flash-latest        | 28.00           |
| gpt-4o                         | 25.00           |
| Llama-3.1-8B                   | 23.00           |
| Meditron3-8B                   | 23.00           |
| Mistral-7Bv0.3                 | 20.00           |
| gpt-4o-mini                    | 15.00           |
| meditron-70B                   |  0.00           |
| medalpaca-13B                  |  0.00           |

For more details on various models and their accuracy across different subjects, please visit our [**Leaderboard**]().

## Contact
- Danilo Bernardo: dbernardoj@gmail.com

## Citation

**BibTeX:**
```bibtex

```
