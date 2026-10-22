# Evaluation suite results

Backend: `neuron`. Completed 18/18 runs; 18 passed.

| Workload | Repeat | Status | Report |
| --- | ---: | --- | --- |
| add | 1 | passed | [JSON](runs/22715f92b50f43718e80797e706447be/result.json) |
| fused_relu | 1 | passed | [JSON](runs/483eb1ca75be46dd979945e831fe77d8/result.json) |
| row_sum | 1 | passed | [JSON](runs/049b919a61b3410cb5b4449082575067/result.json) |
| softmax | 1 | passed | [JSON](runs/aeb1a519363b4a8b8c3e22aebda0561f/result.json) |
| layernorm | 1 | passed | [JSON](runs/47210063959a4866bbde7cfe6e7a2dab/result.json) |
| matmul | 1 | passed | [JSON](runs/21edc2c425434082950c19523dc06573/result.json) |
| fused_relu | 2 | passed | [JSON](runs/9a6e291f22294beca078a83ddff11e6b/result.json) |
| row_sum | 2 | passed | [JSON](runs/6a0bbd2d1b1e483abae9a79c9946c852/result.json) |
| softmax | 2 | passed | [JSON](runs/3186ca6c50594edfb187a75fd8389630/result.json) |
| layernorm | 2 | passed | [JSON](runs/aedf963f768e449d8a4eea19fcd41e81/result.json) |
| matmul | 2 | passed | [JSON](runs/782bdc1b33f84741baa189fe5c551337/result.json) |
| add | 2 | passed | [JSON](runs/870837d9c4e94a019a9e3e90307e0ddd/result.json) |
| row_sum | 3 | passed | [JSON](runs/d3902728762d40988e661c9ed51487f0/result.json) |
| softmax | 3 | passed | [JSON](runs/565d69671bc8443f84470fbf9aa4f115/result.json) |
| layernorm | 3 | passed | [JSON](runs/d619f3a20490499d9991eba56564b40e/result.json) |
| matmul | 3 | passed | [JSON](runs/7412b2ababf94325b150422a9fda1bf6/result.json) |
| add | 3 | passed | [JSON](runs/7467c740a9da4d1c8e84a2d45aa7e410/result.json) |
| fused_relu | 3 | passed | [JSON](runs/a6b43bb945634069b1c30f6901aa72fd/result.json) |

Timing scope: `neuron_host_request_including_transfers`.

| Workload / case | Candidate median ms | Baseline median ms | Median speedup | Repeat range |
| --- | ---: | ---: | ---: | --- |
| add / aligned_fp32 | 0.297303 | 0.301295 | 1.013× | 1.007–1.015× |
| add / bf16 | 0.199769 | 0.196917 | 1.009× | 0.963–1.023× |
| add / tails_fp32 | 0.341765 | 0.304517 | 0.884× | 0.881–0.921× |
| add / zeros | 0.184853 | 0.174106 | 0.952× | 0.928–0.960× |
| fused_relu / mixed_signs | 0.295408 | 0.296631 | 1.005× | 1.004–1.008× |
| fused_relu / negative | 0.193734 | 0.194412 | 1.003× | 0.999–1.006× |
| layernorm / aligned_fp32 | 0.261363 | 0.264594 | 1.012× | 0.994–1.019× |
| layernorm / constant_and_row_tail_fp32 | 0.286752 | 0.259144 | 0.915× | 0.904–0.919× |
| layernorm / small_bf16 | 0.182625 | 0.186236 | 1.020× | 1.001–1.041× |
| matmul / all_dimension_tails_bf16 | 0.226081 | 0.220483 | 0.972× | 0.972–0.975× |
| matmul / multiple_tiles_bf16 | 0.368105 | 0.372391 | 1.011× | 0.992–1.012× |
| matmul / single_tile_bf16 | 0.213631 | 0.220816 | 1.019× | 1.006–1.034× |
| row_sum / aligned_fp32 | 0.225405 | 0.226784 | 1.001× | 0.999–1.006× |
| row_sum / row_and_width_tails_fp32 | 0.260343 | 0.247246 | 0.961× | 0.935–1.057× |
| row_sum / small_bf16 | 0.179874 | 0.178943 | 0.997× | 0.994–0.998× |
| softmax / aligned_fp32 | 0.269065 | 0.261779 | 0.973× | 0.973–0.976× |
| softmax / large_logits_and_row_tail_fp32 | 0.277918 | 0.266807 | 0.974× | 0.908–0.976× |
| softmax / small_bf16 | 0.179399 | 0.180113 | 1.006× | 0.957–1.035× |

Each speedup is the median of paired run ratios, not necessarily the ratio of the displayed medians. The geometric mean weights cases equally. Repeat ranges are descriptive, not confidence intervals. Failed, incomplete, changed-source, mixed-protocol and correctness-only suites have no aggregate performance. Source hashes cover entry files, not imported dependencies.
