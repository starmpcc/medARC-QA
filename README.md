# Medical Abstraction and Reasoning Corpus (MedARC-QA)

[**📖 Paper**](https://arxiv.org/abs/2406.01574) |

This repo contains the evaluation code for the paper "[EVALUATION OF LARGE LANGUAGE MODEL ABSTRACTION AND REASONING IN LONG-TAIL AND OPEN-ENDED MEDICAL TASKS]()"

## Introduction
We introduce MedARC-QA, a question and answer (QA) benchmark designed to evaluate LLMs in long-tail or out-of-distribution medical tasks that are outside their training data corpora. Our results show that a LLMs show poor performance on MedARC-QA contrasting with findings on conventional medical QA (e.g. USMLE), indicating that MMLU-Pro includes more complex reasoning questions. 

## Updates
**January 3, 2025**: Added paper code to the repository.

## Evaluation

To use the API for inference, modify the appropriate API KEY in evaluate_from_api.py script and execute the corresponding bash script:

```bash
cd v1_scripts/
sh eval_gpt_4o.sh
```
## 🏆 Leaderboard
| Model                          | Overall Accuracy |
|--------------------------------|:----------------:|
| gemini-1.5-pro-latest          | 50.00           |
| o1-2024-12-17_200000           | 48.00           |
| Llama-3.1-70B                  | 44.00           |
| claude-3-opus                  | 38.00           |
| Meditron3-8B                   | 36.00           |
| Llama3.1-405b-instruct-fp8     | 34.00           |
| Llama-3.3-70B                  | 34.00           |
| o1-mini                        | 31.00           |
| claude-3-sonnet                | 29.00           |
| gemini-1.5-flash-latest        | 28.00           |
| gpt-4o                         | 25.00           |
| Llama-3.1-8B                   | 23.00           |
| Mistral-7Bv0.3                 | 22.00           |
| gpt-4o-mini                    | 15.00           |
| medalpaca-13b                  |  8.05           |

For more details on various models and their accuracy across different subjects, please visit our [**Leaderboard**]().

## Contact
- Danilo Bernardo: dbernardoj@gmail.com

## Citation

**BibTeX:**
```bibtex

```
