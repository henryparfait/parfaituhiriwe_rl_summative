| Run | Change vs baseline | Hypothesis | Held-out return | Stress return | Treat-rate | Invalid/ep | Frames to 90% | Minutes |
|---|---|---|---|---|---|---|---|---|
| a2c_01_baseline | (baseline) | reference configuration | 9.6 ± 40.9 | 11.1 | 59% | 16.9 | 60000 | 5.0 |
| a2c_02_lr_low | lr=0.00025 | smaller steps | -30.5 ± 21.8 | -31.4 | 40% | 43.5 | 100080 | 5.65 |
| a2c_03_lr_high | lr=0.002 | larger steps: instability? | -7.0 ± 32.8 | 5.5 | 51% | 20.8 | 30000 | 4.42 |
| a2c_04_rollout_32 | frames_per_batch=32 | short rollouts: frequent, noisy updates | 19.6 ± 42.0 | 9.1 | 58% | 15.5 | 45024 | 4.14 |
| a2c_05_rollout_480 | frames_per_batch=480 | long rollouts: fewer, smoother updates | -39.5 ± 23.2 | -30.4 | 45% | 56.3 | 55200 | 3.66 |
| a2c_06_lambda_1 | gae_lambda=1.0 | Monte-Carlo-like advantage: low bias, high variance | -13.3 ± 31.1 | -8.9 | 54% | 43.6 | 45000 | 3.7 |
| a2c_07_lambda_0 | gae_lambda=0.0 | TD(0) advantage: low variance, high bias | 44.4 ± 27.7 | 34.4 | 69% | 1.2 | 40080 | 3.64 |
| a2c_08_no_entropy | entropy_coeff=0.0 | no exploration bonus: early collapse? | 2.9 ± 32.1 | -0.9 | 61% | 24.7 | 60000 | 4.23 |
| a2c_09_entropy_high | entropy_coeff=0.05 | strong exploration bonus | -2.0 ± 38.3 | -4.8 | 58% | 26.0 | 50040 | 4.28 |
| a2c_10_critic_gamma | gamma=0.95, critic_coeff=1.0 | stronger critic, shorter horizon | 2.3 ± 29.5 | 6.4 | 57% | 22.8 | 60000 | 4.08 |
| a2c_11_wide | hidden=256 | wider networks | 22.4 ± 29.3 | 5.7 | 63% | 6.8 | 85080 | 3.81 |
| a2c_07_lambda_0_long | gae_lambda=0.0 | best 100k config (a2c_07_lambda_0) retrained for 300,000 frames | 47.6 ± 39.6 | 26.1 | 62% | 1.3 | 40080 | 13.29 |
