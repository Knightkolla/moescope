import numpy as np
import pytest

torch = pytest.importorskip("torch")
T = pytest.importorskip("transformers")
import moescope  # noqa: E402
from moescope.monitor import MoescopeCallback, RoutingMonitor, layer_counts, window_stats  # noqa: E402


def toy():
    torch.manual_seed(0)
    cfg = T.OlmoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=4, num_experts=16, num_experts_per_tok=4,
                        max_position_embeddings=128, pad_token_id=0)
    return T.OlmoeForCausalLM(cfg)


def test_window_stats():
    log = np.zeros(6, moescope.RECORD_DTYPE)
    log["layer_id"] = [0, 0, 0, 0, 1, 1]
    log["expert_id"] = [0, 0, 1, 3, 2, 2]
    log["flags"] = [0, 0, 0, moescope.CAPACITY_DROP, 0, 0]
    stats, loads = window_stats(layer_counts(log, 4), 4)
    assert stats["moescope/layer0/imbalance"] == pytest.approx(4 * 2 / 3)
    assert stats["moescope/layer0/dead_experts"] == 2  # expert 3 only had a dropped decision
    assert stats["moescope/layer0/capacity_drop_rate"] == 0.25
    assert stats["moescope/layer1/entropy"] == 0.0
    stats2, _ = window_stats(layer_counts(log, 4), 4, previous=loads)
    assert stats2["moescope/mean/drift_js"] == 0.0


def test_training_with_gradient_checkpointing_logs_each_token_once():
    model = toy().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    seen = []
    session = moescope.attach(model, device="cpu", capacity=10_000)
    mon = RoutingMonitor(session, every=2, log_fn=lambda s, step: seen.append((step, s)))
    for step in range(4):
        ids = torch.randint(3, 256, (2, 16))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        mon.step(step)
    session.detach()
    assert [s for s, _ in seen] == [1, 3]
    for _, stats in seen:  # 2 steps x 32 tokens x 2 layers x k=4, not doubled by recomputation
        assert stats["moescope/decisions"] == 2 * 32 * 2 * 4
        assert stats["moescope/records_lost"] == 0
    assert "moescope/mean/drift_js" in seen[1][1]


def test_trainer_callback():
    pytest.importorskip("accelerate")
    model = toy()
    data = [{"input_ids": torch.randint(3, 256, (16,)), "labels": torch.randint(3, 256, (16,))} for _ in range(8)]
    cb = MoescopeCallback(every=2, device="cpu")
    args = T.TrainingArguments(output_dir="/tmp/moescope-trainer-test", per_device_train_batch_size=2, max_steps=4,
                               report_to=[], save_strategy="no", use_cpu=True, logging_steps=100)
    T.Trainer(model=model, args=args, train_dataset=data, callbacks=[cb]).train()
    assert len(cb.history) == 2 and cb.history[0]["moescope/decisions"] == 2 * 2 * 16 * 2 * 4
