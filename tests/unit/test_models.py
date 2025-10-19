from __future__ import annotations

from src.models.model_manager import ModelManager


def test_model_manager_stub_inference():
    configs = [
        {
            "name": "vision",
            "class": "models.torch_model.TorchModel",
            "enabled": True,
            "options": {},
        }
    ]
    manager = ModelManager(configs)
    manager.load_enabled()

    result = manager.infer("vision", {"data": [1, 2, 3]})

    assert result["model"] == "vision"
    assert "confidence" in result

    manager.unload_all()
