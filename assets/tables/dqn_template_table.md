| Run | Learning Rate | Gamma | Replay Buffer Size | Batch Size | Exploration Strategy | Target Update (steps) | Other | Mean Reward |
|---|---|---|---|---|---|---|---|---|
| dqn_01_baseline | 0.0005 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 0.25 | 9.3 ± 23.9 |
| dqn_02_lr_low | 0.0001 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 0.25 | 41.1 ± 36.7 |
| dqn_03_lr_high | 0.002 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 0.25 | -28.4 ± 15.1 |
| dqn_04_gamma_090 | 0.0005 | 0.9 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 0.25 | 66.7 ± 43.9 |
| dqn_05_small_buffer | 0.0005 | 0.99 | 5000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 0.25 | 25.2 ± 30.0 |
| dqn_06_batch_256 | 0.0005 | 0.99 | 50000 | 256 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 0.25 | 47.4 ± 32.2 |
| dqn_07_fast_eps | 0.0005 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.01 over 10% of frames | 500 | net 128x2, UTD 0.25 | 52.4 ± 32.9 |
| dqn_08_slow_eps | 0.0005 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.1 over 80% of frames | 500 | net 128x2, UTD 0.25 | -14.4 ± 18.1 |
| dqn_09_target_fast | 0.0005 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 50 | net 128x2, UTD 0.25 | -31.8 ± 26.3 |
| dqn_10_target_slow | 0.0005 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 2000 | net 128x2, UTD 0.25 | 49.3 ± 31.3 |
| dqn_11_double_wide | 0.0005 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | Double DQN, net 256x2, UTD 0.25 | 13.3 ± 32.9 |
| dqn_12_high_utd | 0.0005 | 0.99 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 1.0 | 5.5 ± 27.8 |
| dqn_04_gamma_090_long | 0.0005 | 0.9 | 50000 | 64 | ε-greedy 1.0→0.05 over 40% of frames | 500 | net 128x2, UTD 0.25 | 90.3 ± 34.4 |
