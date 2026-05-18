# Schedule-Free 8bit fork

Schedule-Free Optimizers in PyTorch.

Original Preprint: [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)

Authors: Aaron Defazio, Xingyu (Alice) Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky

**TLDR** Faster training without schedules - no need to specify the stopping time/steps in advance!

We provide several Schedule-Free optimizer implementations:

- Experimental `AdamWScheduleFree8bit`: Schedule-free AdamW with the Adam second-moment state stored in block-wise 8-bit form
- Experimental `RAdamScheduleFree8bit`: Schedule-free RAdam with the RAdam second-moment state stored in block-wise 8-bit form

