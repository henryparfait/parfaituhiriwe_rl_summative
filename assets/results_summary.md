# Results summary

Baselines (30 held-out episodes): heuristic 73.6 (treat-rate 71%), random -11.2

| Algorithm | Best run | Held-out return | Stress return | Precancer treat-rate | Precancers lost | Invalid actions | Frames to 90% |
|---|---|---|---|---|---|---|---|
| DQN | dqn_04_gamma_090 | 66.7 ± 43.9 | 50.2 | 68% | 1.67 | 5.4 | 55000 |
| REINFORCE | reinforce_07_batch_2 | -49.0 ± 15.6 | -51.1 | 24% | 2.57 | 49.8 | 100185 |
| A2C | a2c_07_lambda_0 | 44.4 ± 27.7 | 34.4 | 69% | 2.00 | 1.2 | 40080 |
| PPO | ppo_06_epochs_10 | 58.2 ± 33.9 | 45.7 | 69% | 1.53 | 0.6 | 50176 |

Episodes to reach 90% of final evaluation level (best run): DQN 482, REINFORCE 846, A2C 370, PPO 423

| Extended run | Held-out return | Stress return | Treat-rate |
|---|---|---|---|
| dqn_04_gamma_090_long | 90.3 ± 34.4 | 62.6 | 75% |
| reinforce_07_batch_2_long | 15.1 ± 37.9 | 11.3 | 61% |
| a2c_07_lambda_0_long | 47.6 ± 39.6 | 26.1 | 62% |
| ppo_06_epochs_10_long | 102.3 ± 30.3 | 86.4 | 81% |
