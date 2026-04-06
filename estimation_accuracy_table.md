| benchmark | Qwen32B-Think | Qwen32B-Instant | GPT5.2-Think | GPT5.2-Instant | multi-turn |
| --- | --- | --- | --- | --- | --- |
| sokoban | turn err 71.85%<br>token err 98.48% | turn err 81.45%<br>token err 99.23% | turn err 29.68%<br>token err 27.24% | turn err 825.24%<br>token err 29.02% | yes |
| webshop | turn err 48.89%<br>token err 44.81% | turn err 41.48%<br>token err 35.80% | turn err 12.10%<br>token err 17.14% | turn err 34.34%<br>token err 26.19% | yes |
| frozen-lake | turn err 66.42%<br>token err 99.28% | turn err 40.04%<br>token err 99.11% | turn err 37.09%<br>token err 22.55% | turn err 20.36%<br>token err 36.59% | yes |
| deepcoder | token err 258.53% | token err 48.48% | token err 27.26% | token err 38.40% | no |
| search-r1 | turn err 48.97%<br>token err 43.00% | turn err 44.15%<br>token err 59.73% | turn err 33.63%<br>token err 24.32% | turn err 34.11%<br>token err 34.95% | yes |
| gpqa-main | token err 68.51% | token err 73.22% | token err 33.26% | token err 52.01% | no |

Relative error = mean(|estimate - actual| / max(1, actual)) over valid turns.