| benchmark | Qwen32B-Think | Qwen32B-Instant | GPT5.2-Think | GPT5.2-Instant | multi-turn |
| --- | --- | --- | --- | --- | --- |
| sokoban | turn 46.66%<br>token 1.52% | turn 46.73%<br>token 0.77% | turn 76.68%<br>token 73.00% | turn 59.58%<br>token 70.98% | yes |
| webshop | turn 56.56%<br>token 55.59% | turn 59.11%<br>token 64.20% | turn 88.13%<br>token 82.87% | turn 70.62%<br>token 73.81% | yes |
| frozen-lake | turn 53.31%<br>token 0.72% | turn 67.63%<br>token 0.89% | turn 69.65%<br>token 77.45% | turn 80.46%<br>token 63.42% | yes |
| deepcoder | 53.78% | 57.08% | 73.08% | 63.60% | no |
| search-r1 | turn 54.05%<br>token 57.03% | turn 56.51%<br>token 40.28% | turn 74.49%<br>token 75.68% | turn 77.67%<br>token 65.05% | yes |
| gpqa-main | 34.48% | 30.45% | 66.93% | 48.78% | no |

Accuracy = mean(max(0, 1 - |estimate - actual| / max(1, actual))) over valid turns.