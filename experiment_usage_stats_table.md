| benchmark | model | mean token | p70 token | p30 token | mean turn | p70 turn | p30 turn | multi-turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sokoban | Qwen32B-Think | N/A | N/A | N/A | 9.98 | 10.00 | 10.00 | yes |
| sokoban | Qwen32B-Instant | N/A | N/A | N/A | 10.00 | 10.00 | 10.00 | yes |
| sokoban | GPT5.2-Think | N/A | N/A | N/A | 6.13 | 8.00 | 4.00 | yes |
| sokoban | GPT5.2-Instant | N/A | N/A | N/A | 6.80 | 10.00 | 5.00 | yes |
| webshop | Qwen32B-Think | N/A | N/A | N/A | 8.54 | 9.00 | 9.00 | yes |
| webshop | Qwen32B-Instant | N/A | N/A | N/A | 9.00 | 9.00 | 9.00 | yes |
| webshop | GPT5.2-Think | 13786.05 | 14056.60 | 11285.20 | 5.27 | 5.00 | 5.00 | yes |
| webshop | GPT5.2-Instant | 10422.62 | 11001.50 | 10419.60 | 4.75 | 5.00 | 5.00 | yes |
| frozen-lake | Qwen32B-Think | N/A | N/A | N/A | 10.00 | 10.00 | 10.00 | yes |
| frozen-lake | Qwen32B-Instant | N/A | N/A | N/A | 10.00 | 10.00 | 10.00 | yes |
| frozen-lake | GPT5.2-Think | 4511.48 | 4863.60 | 1824.40 | 3.90 | 4.00 | 2.00 | yes |
| frozen-lake | GPT5.2-Instant | 2333.75 | 3072.50 | 1185.90 | 2.83 | 4.00 | 2.00 | yes |
| deepcoder | Qwen32B-Think | N/A | N/A | N/A | 1.00 | 1.00 | 1.00 | no |
| deepcoder | Qwen32B-Instant | N/A | N/A | N/A | 1.00 | 1.00 | 1.00 | no |
| deepcoder | GPT5.2-Think | 1575.77 | 1932.80 | 1088.40 | 1.00 | 1.00 | 1.00 | no |
| deepcoder | GPT5.2-Instant | 1458.13 | 1774.00 | 1048.30 | 1.00 | 1.00 | 1.00 | no |
| search-r1 | Qwen32B-Think | N/A | N/A | N/A | 4.80 | 5.00 | 5.00 | yes |
| search-r1 | Qwen32B-Instant | N/A | N/A | N/A | 5.00 | 5.00 | 5.00 | yes |
| search-r1 | GPT5.2-Think | 2385.37 | 2477.70 | 1531.60 | 2.43 | 2.00 | 2.00 | yes |
| search-r1 | GPT5.2-Instant | 1438.25 | 1722.70 | 1011.00 | 2.45 | 3.00 | 2.00 | yes |
| gpqa-main | Qwen32B-Think | N/A | N/A | N/A | 1.00 | 1.00 | 1.00 | no |
| gpqa-main | Qwen32B-Instant | N/A | N/A | N/A | 1.00 | 1.00 | 1.00 | no |
| gpqa-main | GPT5.2-Think | N/A | N/A | N/A | 1.00 | 1.00 | 1.00 | no |
| gpqa-main | GPT5.2-Instant | N/A | N/A | N/A | 1.00 | 1.00 | 1.00 | no |

Each row is computed over the latest trajectory for each unique env_id.
p70 / p30 mean the 70th / 30th percentile of per-env token usage or turn count.