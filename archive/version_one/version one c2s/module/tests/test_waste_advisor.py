import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from waste_advisor import WasteAdvisor

class TestWasteAdvisor(unittest.TestCase):
    def setUp(self):
        self.bot = WasteAdvisor()

    def test_plastic_queries(self):
        resp = self.bot.get_response("What is PET plastic?")
        self.assertIn("Polyethylene Terephthalate", resp["message"])
        self.assertIn("#1", resp["message"])

    def test_decomposition_queries(self):
        resp = self.bot.get_response("How long does styrofoam take to decompose?")
        self.assertIn("500+ years", resp["message"])

    def test_composting_queries(self):
        resp = self.bot.get_response("How to start composting?")
        self.assertIn("Composting", resp["message"])

    def test_glass_queries(self):
        resp = self.bot.get_response("Can glass be recycled?")
        self.assertIn("Glass", resp["message"])

    def test_help_command(self):
        resp = self.bot.get_response("help")
        self.assertEqual(resp["type"], "help")

    def test_greeting(self):
        resp = self.bot.get_response("hello")
        self.assertEqual(resp["type"], "greeting")

    def test_uploaded_item_reuse_without_detection(self):
        resp = self.bot.get_response("how the item can be reused that i uploaded")
        self.assertEqual(resp["type"], "no_detection_context")
        self.assertIn("No waste item has been uploaded", resp["message"])

    def test_uploaded_item_reuse_with_detection(self):
        self.bot.set_last_detection({"waste_type": "plastic_bottle", "confidence": 0.95})
        resp = self.bot.get_response("how the item can be reused that i uploaded")
        self.assertIn(resp["type"], ("detection_reuse", "household_reuse"))
        self.assertIn("Reuse", resp["message"])
        self.assertIn("Planter", resp["message"])

    def test_verb_can_not_triggering_metal(self):
        resp = self.bot.get_response("How can plastic be reused?")
        self.assertEqual(resp["type"], "reuse")
        self.assertIn("Plastic", resp["message"])
        self.assertNotIn("Aluminium cans", resp["message"])

    def test_user_exact_question_guess_recycle_household_reuse(self):
        # When an item was detected as plastic bottle (h/w > 1.3)
        self.bot.set_last_detection({
            "waste_type": "plastic",
            "confidence": 0.92,
            "bbox": [100, 50, 200, 300]  # height = 250, width = 100 -> ratio 2.5 (bottle)
        })
        q = "guess what object it be and how the object can be recycle how it can be reused for household purpose"
        resp = self.bot.get_response(q)
        self.assertEqual(resp["type"], "complete_detection_analysis")
        # Check that it guessed the object
        self.assertIn("Single-Use Plastic Bottle", resp["message"])
        self.assertIn("PET", resp["message"])
        # Check that it tells how to recycle
        self.assertIn("How the Object Can Be Recycled", resp["message"])
        self.assertIn("Crush the bottle body flat", resp["message"])
        # Check that it tells how to reuse for household purpose
        self.assertIn("Reused for Household Purpose", resp["message"])
        self.assertIn("Self-Watering Planter", resp["message"])

    def test_household_purpose_specific_query(self):
        self.bot.set_last_detection({
            "waste_type": "plastic",
            "confidence": 0.88,
            "bbox": [100, 100, 300, 300]  # squarish -> container
        })
        resp = self.bot.get_response("how can it be reused for household purpose")
        self.assertEqual(resp["type"], "household_reuse")
        self.assertIn("Household Reuse", resp["message"])
        self.assertIn("Hardware & Workshop Bins", resp["message"])

    def test_guess_object_and_analysis_method(self):
        analysis = self.bot.guess_object_and_analysis("plastic", 0.9, bbox=[0, 0, 100, 250])
        self.assertIn("Bottle", analysis["guessed_object"])
        self.assertIn("PET", analysis["plastic_type"])
        self.assertTrue(len(analysis["recycling_steps"]) >= 3)
        self.assertTrue(len(analysis["household_reuse"]) >= 4)

    def test_plastic_chair_detection_analysis(self):
        analysis = self.bot.guess_object_and_analysis("plastic", 0.85, detected_object="Plastic Chair / Furniture")
        self.assertIn("Plastic Chair", analysis["guessed_object"])
        self.assertIn("PP #5", analysis["plastic_type"])
        self.assertTrue(len(analysis["recycling_steps"]) >= 3)
        self.assertTrue(len(analysis["household_reuse"]) >= 4)
        self.assertIn("Plant Stand", analysis["household_reuse"][0])

    def test_user_is_it_can_be_reusable(self):
        self.bot.set_last_detection({"waste_type": "plastic", "confidence": 0.82})
        resp = self.bot.get_response("is it can be reusable")
        self.assertEqual(resp["type"], "household_reuse")
        self.assertIn("Yes, absolutely!", resp["message"])
        self.assertIn("Household Reuse", resp["message"])

    def test_user_will_it_use_for_another_purpose(self):
        self.bot.set_last_detection({"waste_type": "plastic", "confidence": 0.85})
        resp = self.bot.get_response("will it use for another purpose")
        self.assertEqual(resp["type"], "household_reuse")
        self.assertIn("Yes, absolutely!", resp["message"])
        self.assertIn("Household Reuse", resp["message"])

    def test_knowledge_encyclopedia_burn_plastic(self):
        resp = self.bot.get_response("can we burn plastic at home")
        self.assertEqual(resp["type"], "web_knowledge_summary")
        self.assertIn("Burning Plastic", resp["message"])
        self.assertIn("Never burn plastic", resp["message"])

    def test_knowledge_encyclopedia_pizza_box(self):
        resp = self.bot.get_response("what to do with pizza box")
        self.assertEqual(resp["type"], "web_knowledge_summary")
        self.assertIn("Pizza Boxes", resp["message"])
        self.assertIn("cardboard", resp["message"])

if __name__ == "__main__":
    unittest.main()
