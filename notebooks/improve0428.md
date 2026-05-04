Do not use average pooling for the 1x8 bottleneck
Use a learned global encoder:
x[batch,1,L] -> MLP/Conv stack -> flatten -> Linear -> [batch,1,8].
This at least lets each latent scalar encode a learned global feature, not just a coarse spatial average.

Use an MLP latent dynamics, not local Conv1d
With only 8 latent positions, locality in latent space is artificial. A small residual MLP over the 8 numbers is probably better:
[batch,1,8] -> flatten -> MLP -> [batch,1,8].

Use a learned decoder, not interpolation
Current decoder upsamples 8 values to 128 by smooth interpolation, then local conv. That biases outputs toward smooth coarse patterns, while CA states are high-frequency binary patterns. Better:
latent[8] -> MLP -> logits[128], optionally with positional embeddings.

Pretrain reconstruction first
Before asking it to learn dynamics, make sure the 1x8 autoencoder can reconstruct x_t. If reconstruction hamming is bad, dynamics has no chance. Use high recon_weight, train only autoencoding for a stage, then add rollout.

Use a horizon curriculum
Start rollout horizon at 1, then 2, 4, 8, 16. Jumping straight to horizon 16 with this bottleneck is harsh.

Re-enable a teacher/closure latent loss
The run used pure latent rollout, but with such a small latent, I’d add closure_weight so step_latent(encode(x_t)) is explicitly pulled toward encode(x_{t+1}).