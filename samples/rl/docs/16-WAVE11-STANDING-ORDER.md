# Wave 11 standing order — **SUPERSEDED**

**Superseded by** [`16-JOINT-PRETRAIN-STANDING.md`](./16-JOINT-PRETRAIN-STANDING.md) and [`07-TRAINING-PLAYBOOK.md`](./07-TRAINING-PLAYBOOK.md).

The Wave 11 order to ship standalone `bc_scripted.pt` and merge with a separate critic is **invalid** under the same-expert mandate:

1. BC and critic pretrain **together on the same expert**.  
2. **Do not queue** until readiness gates pass.

Historical content below is frozen for archaeology only — do not follow for Job1.

---

<details>
<summary>Historical Wave 11 text (do not use)</summary>

While GPU is blocked, produce `bc_scripted.pt` on CPU, optionally merge with `critic_pretrained.pt`, curriculum empty, W7N3. That product path is replaced by `pretrain_joint` → `joint_pretrain.pt`.

</details>
