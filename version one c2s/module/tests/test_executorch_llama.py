#!/usr/bin/env python3
"""
Unit tests for ExecuTorch Llama Engine.
Smart Waste Segregation System - Bharat AI-SoC Challenge PS2
"""

import unittest
from executorch_llama_engine import ExecuTorchLlamaEngine
from config import LlamaConfig
from waste_advisor import WasteAdvisor


class TestExecuTorchLlamaEngine(unittest.TestCase):

    def setUp(self):
        self.config = LlamaConfig(enabled=True)
        self.engine = ExecuTorchLlamaEngine(self.config)

    def test_engine_initialization(self):
        result = self.engine.initialize()
        self.assertTrue(result)
        self.assertTrue(self.engine.model_loaded)

    def test_prompt_formatting(self):
        prompt = self.engine.format_llama3_prompt("How to recycle plastic?", context="Plastic bottle detected")
        self.assertIn("<|begin_of_text|>", prompt)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("Plastic bottle detected", prompt)
        self.assertIn("How to recycle plastic?", prompt)

    def test_generation_output(self):
        self.engine.initialize()
        response = self.engine.generate("How do I recycle a bottle?")
        self.assertIn("text", response)
        self.assertIn("engine", response)
        self.assertTrue(response["on_device"])
        self.assertGreaterEqual(response["inference_time_ms"], 0.0)

    def test_waste_advisor_integration(self):
        advisor = WasteAdvisor()
        self.assertIsNotNone(advisor.llama_engine)
        res = advisor.get_response("Give me advice on microplastics")
        self.assertIsNotNone(res)
        self.assertIn("message", res)


if __name__ == "__main__":
    unittest.main()
