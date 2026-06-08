import pytest

from src.training.trainer import TrainingConfig


class TestTrainingConfig:
    def test_creates_with_defaults(self):
        config = TrainingConfig(data="LLM/data")
        assert config.data == "LLM/data"
        assert config.iters == 600
        assert config.batch_size == 4
        assert config.backend == "auto"
        assert config.model == "unsloth/gemma-3-1b-it"
        assert config.chat_template is None
        assert config.learning_rate == 2e-4
        assert config.lora_rank == 64
        assert config.max_seq_length == 2048

    def test_creates_with_custom_values(self):
        config = TrainingConfig(
            data="/tmp/data",
            iters=100,
            batch_size=2,
            backend="cuda",
            model="test-model",
            chat_template="gemma",
            learning_rate=1e-4,
            lora_rank=32,
            max_seq_length=1024,
        )
        assert config.data == "/tmp/data"
        assert config.iters == 100
        assert config.batch_size == 2
        assert config.backend == "cuda"
        assert config.model == "test-model"
        assert config.chat_template == "gemma"
        assert config.learning_rate == 1e-4
        assert config.lora_rank == 32
        assert config.max_seq_length == 1024

    def test_raises_on_empty_data(self):
        with pytest.raises(ValueError, match="data path is required"):
            TrainingConfig(data="")

    def test_raises_on_negative_iters(self):
        with pytest.raises(ValueError, match="iters must be positive"):
            TrainingConfig(data="LLM/data", iters=0)

    def test_raises_on_negative_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(data="LLM/data", batch_size=0)

    def test_raises_on_negative_lora_rank(self):
        with pytest.raises(ValueError, match="lora_rank must be positive"):
            TrainingConfig(data="LLM/data", lora_rank=0)

    def test_raises_on_negative_max_seq_length(self):
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            TrainingConfig(data="LLM/data", max_seq_length=0)

    def test_raises_on_invalid_backend(self):
        with pytest.raises(ValueError, match="backend must be"):
            TrainingConfig(data="LLM/data", backend="invalid")

    def test_accepts_valid_backends(self):
        for backend in ("auto", "cuda", "mlx"):
            config = TrainingConfig(data="LLM/data", backend=backend)
            assert config.backend == backend
