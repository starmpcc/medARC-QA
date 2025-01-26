# Medical Abstraction and Reasoning Corpus (MedARC-QA)

[**📖 Paper**]() |

This repo contains the evaluation code for the paper "[Limitiations of Large Language Models in Medical Reasoning Arising from Cognitive Rigidity]()"

## Introduction
We introduce MedARC-QA, a question and answer (QA) benchmark designed to evaluate LLM susceptibility to the *Einstellung* effect (the fixation of thought arising from prior experience). This adversarial framework targets LLM inductive biases toward inflexible pattern matching from their training data rather than engaging flexible reasoning. Our results show that a LLMs show poor performance on MedARC-QA contrasting with findings on conventional medical QA (e.g. USMLE). 

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

## 🏆 Leaderboard (Model Cut-off Date 12/17/2024)
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
