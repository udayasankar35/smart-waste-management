#!/usr/bin/env python3
"""
Waste Advisory Chatbot Module
AI-Powered Smart Waste Classification & Disposal Assistant

Provides intelligent waste disposal recommendations, recycling guidance,
and environmental impact education. Rule-based with comprehensive
knowledge base and fuzzy NLP matching.

Bharat AI-SoC Challenge 2026–27 — PS2
"""

import re
import time
import json
import urllib.request
import urllib.parse
import logging
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

logger = logging.getLogger(__name__)


# ─── Plastic Types Knowledge ─────────────────────────────────────────────────────

PLASTIC_TYPES = {
    "pet": {
        "code": "#1 — PET (Polyethylene Terephthalate)",
        "common_items": "water bottles, soft drink bottles, food jars, polyester clothing",
        "recyclable": True,
        "recycled_into": "polyester fibre, new bottles, carpet, fleece jackets",
        "decomposition": "450 years",
        "health_risk": "Safe for single use. Do NOT reuse — can leach antimony and BPA-like chemicals when heated.",
        "tip": "Crush and cap before recycling. Rinse out food residue.",
    },
    "hdpe": {
        "code": "#2 — HDPE (High-Density Polyethylene)",
        "common_items": "milk jugs, detergent bottles, shampoo bottles, grocery bags",
        "recyclable": True,
        "recycled_into": "pipes, plastic lumber, playground equipment, recycling bins",
        "decomposition": "500 years",
        "health_risk": "Considered one of the safest plastics. OK to reuse.",
        "tip": "Most curbside programs accept HDPE. Rinse containers before recycling.",
    },
    "pvc": {
        "code": "#3 — PVC (Polyvinyl Chloride)",
        "common_items": "pipes, window frames, shower curtains, credit cards, blister packs",
        "recyclable": False,
        "recycled_into": "flooring, paneling, speed bumps (limited facilities)",
        "decomposition": "Up to 1000 years",
        "health_risk": "⚠️ Contains chlorine and phthalates. Can release dioxins when burned. Avoid for food.",
        "tip": "Very difficult to recycle. Avoid when possible. Never burn PVC.",
    },
    "ldpe": {
        "code": "#4 — LDPE (Low-Density Polyethylene)",
        "common_items": "plastic wraps, squeeze bottles, bread bags, frozen food bags",
        "recyclable": True,
        "recycled_into": "bin liners, floor tiles, shipping envelopes, furniture",
        "decomposition": "500 years",
        "health_risk": "Relatively safe. Does not leach harmful chemicals.",
        "tip": "Many grocery stores have drop-off bins for LDPE bags and wraps.",
    },
    "pp": {
        "code": "#5 — PP (Polypropylene)",
        "common_items": "yogurt cups, ketchup bottles, straws, bottle caps, medicine bottles",
        "recyclable": True,
        "recycled_into": "automotive parts, signal lights, ice scrapers, rakes, textiles",
        "decomposition": "20–30 years",
        "health_risk": "Safe for food use. Microwave-safe. One of the best plastics.",
        "tip": "PP is increasingly accepted in recycling. Check local rules for caps.",
    },
    "ps": {
        "code": "#6 — PS (Polystyrene / Styrofoam)",
        "common_items": "disposable cups, takeout containers, packing peanuts, insulation",
        "recyclable": False,
        "recycled_into": "insulation, license plate frames, rulers (very rarely recycled)",
        "decomposition": "500+ years",
        "health_risk": "⚠️ Can leach styrene (a possible carcinogen) especially when heated. Avoid for hot food/drinks.",
        "tip": "Avoid styrofoam whenever possible. It breaks into microplastics that harm marine life.",
    },
    "other": {
        "code": "#7 — Other (Polycarbonate, Nylon, ABS, etc.)",
        "common_items": "baby bottles (old), CDs/DVDs, sunglasses, phone cases, 3D prints",
        "recyclable": False,
        "recycled_into": "Generally not recyclable through curbside programs",
        "decomposition": "Indefinite — varies widely",
        "health_risk": "⚠️ May contain BPA. Avoid polycarbonate (#7-PC) for food and beverages.",
        "tip": "If marked 'PLA' or 'compostable', it's plant-based and can be industrially composted.",
    },
}


# ─── Waste Knowledge Base ────────────────────────────────────────────────────────

WASTE_KNOWLEDGE = {
    "plastic": {
        "category": "Recyclable — Plastic",
        "icon": "🛢️",
        "color": "#FF9800",
        "disposal": [
            "Rinse the item to remove food residue.",
            "Remove labels and caps if possible.",
            "Place in the BLUE recycling bin (dry waste).",
            "Do NOT mix with wet waste or food scraps.",
            "Crush bottles to save space in the bin.",
        ],
        "recycling": [
            "PET (#1) — bottles, containers → remade into polyester fibre, new bottles.",
            "HDPE (#2) — milk jugs, detergent → pipes, lumber, playground equipment.",
            "PVC (#3) — pipes, cards → flooring, cables (limited recycling).",
            "LDPE (#4) — bags, wraps → bin liners, furniture.",
            "PP (#5) — yogurt cups, straws → automotive parts, textiles.",
            "PS (#6) — styrofoam → insulation, frames (rarely recycled).",
        ],
        "impact": {
            "decomposition_years": "450–1000 years",
            "co2_saved_kg": 1.5,
            "water_saved_liters": 16.3,
            "energy_saved_kwh": 5.77,
            "fact": "Recycling 1 plastic bottle saves enough energy to power a 60W bulb for 6 hours.",
        },
        "tips": [
            "Use reusable water bottles and bags to reduce plastic waste.",
            "Avoid single-use plastics like straws and cutlery.",
            "Choose products with minimal packaging.",
            "Support local recycling programmes.",
        ],
        "hazard_level": "low",
    },
    "recyclable": {
        "category": "Recyclable — Paper/Cardboard/Glass",
        "icon": "♻️",
        "color": "#4CAF50",
        "disposal": [
            "Flatten cardboard boxes before recycling.",
            "Keep paper clean and dry — wet paper cannot be recycled.",
            "Rinse glass bottles and jars.",
            "Place in the BLUE recycling bin.",
            "Remove tape, staples, and plastic windows from envelopes.",
        ],
        "recycling": [
            "Paper → new paper products (can be recycled 5-7 times).",
            "Cardboard → new packaging, paperboard.",
            "Glass → new bottles and jars (infinitely recyclable).",
            "Newspaper → newsprint, insulation, animal bedding.",
        ],
        "impact": {
            "decomposition_years": "Paper: 2-6 weeks | Glass: 1 million years",
            "co2_saved_kg": 0.9,
            "water_saved_liters": 26.5,
            "energy_saved_kwh": 4.0,
            "fact": "Recycling 1 tonne of paper saves 17 trees and 26,500 litres of water.",
        },
        "tips": [
            "Print double-sided to reduce paper waste.",
            "Use digital documents when possible.",
            "Compost non-glossy paper if recycling isn't available.",
        ],
        "hazard_level": "low",
    },
    "metal": {
        "category": "Recyclable — Metal",
        "icon": "🔩",
        "color": "#607D8B",
        "disposal": [
            "Rinse food cans before recycling.",
            "Aluminium foil: clean and ball up before recycling.",
            "Place in the BLUE recycling bin.",
            "Large metal items → take to scrap metal dealer or collection centre.",
            "Do NOT dispose of aerosol cans that are not fully empty.",
        ],
        "recycling": [
            "Aluminium cans → new cans in just 60 days (infinitely recyclable).",
            "Steel/tin cans → car parts, appliances, construction materials.",
            "Copper → electrical wiring, plumbing.",
            "Scrap metal → melted and remanufactured.",
        ],
        "impact": {
            "decomposition_years": "50–200 years",
            "co2_saved_kg": 9.0,
            "water_saved_liters": 40.0,
            "energy_saved_kwh": 14.0,
            "fact": "Recycling aluminium uses 95% less energy than producing new aluminium.",
        },
        "tips": [
            "Collect aluminium cans — they have high scrap value.",
            "Separate ferrous and non-ferrous metals.",
            "Check if your local kabadiwala accepts metal scrap.",
        ],
        "hazard_level": "low",
    },
    "dry_waste": {
        "category": "Dry Waste — Non-Recyclable",
        "icon": "🍂",
        "color": "#795548",
        "disposal": [
            "Place in the GREEN dry-waste bin.",
            "Keep separate from wet waste and recyclables.",
            "Includes: fabric scraps, rubber, ceramic, soiled paper.",
            "Multi-layered packaging (chips packets, tetra paks) goes here.",
            "Sanitary waste: wrap securely before disposal.",
        ],
        "recycling": [
            "Most dry waste goes to sanitary landfill.",
            "Some items can be upcycled: old clothes → cleaning rags.",
            "Tetra paks: some facilities accept them — check locally.",
            "Consider waste-to-energy conversion for non-recyclable dry waste.",
        ],
        "impact": {
            "decomposition_years": "Varies — weeks to centuries",
            "co2_saved_kg": 0.3,
            "water_saved_liters": 5.0,
            "energy_saved_kwh": 1.0,
            "fact": "Proper segregation can divert up to 80% of dry waste from landfills.",
        },
        "tips": [
            "Reduce consumption of single-use items.",
            "Donate usable clothes and textiles.",
            "Compost paper towels and tissues instead of binning.",
        ],
        "hazard_level": "low",
    },
    "wet_waste": {
        "category": "Wet Waste — Organic / Biodegradable",
        "icon": "💧",
        "color": "#8BC34A",
        "disposal": [
            "Place in the GREEN wet-waste bin.",
            "Includes: food scraps, fruit peels, vegetable waste, tea bags.",
            "Do NOT mix with plastic, glass, or dry waste.",
            "Wrap in newspaper to contain moisture.",
            "Drain excess liquid before disposal.",
        ],
        "recycling": [
            "Composting → nutrient-rich soil for gardening.",
            "Vermicomposting → worm-driven composting for apartments.",
            "Biogas generation → methane fuel from organic waste.",
            "Bokashi method → fermented composting for all food waste.",
        ],
        "impact": {
            "decomposition_years": "1-6 months",
            "co2_saved_kg": 0.5,
            "water_saved_liters": 10.0,
            "energy_saved_kwh": 0.8,
            "fact": "Food waste in landfills produces methane, a greenhouse gas 28x more potent than CO₂.",
        },
        "tips": [
            "Start a kitchen compost bin.",
            "Plan meals to reduce food waste.",
            "Freeze food scraps for stock-making.",
            "Use banana peels and eggshells as plant fertiliser.",
        ],
        "hazard_level": "low",
    },
    "hazardous": {
        "category": "Hazardous Waste — Requires Special Handling",
        "icon": "☠️",
        "color": "#F44336",
        "disposal": [
            "⚠️ Do NOT place in regular bins!",
            "Store safely in original containers away from children.",
            "Take to an authorised hazardous waste collection facility.",
            "Batteries: collect and take to designated battery drop-off points.",
            "Chemicals/paint: never pour down drains. Contact local authority for disposal.",
            "Medical waste: use designated biomedical waste bins (yellow bags).",
        ],
        "recycling": [
            "Batteries → specialised recycling (lead, lithium recovered).",
            "E-waste → certified e-waste recyclers extract valuable metals.",
            "Motor oil → re-refined into new lubricants.",
            "Fluorescent bulbs → mercury recovery.",
        ],
        "impact": {
            "decomposition_years": "Never — toxic compounds persist",
            "co2_saved_kg": 0.0,
            "water_saved_liters": 0.0,
            "energy_saved_kwh": 0.0,
            "fact": "Improper hazardous waste disposal can contaminate soil and groundwater for decades.",
        },
        "tips": [
            "Use eco-friendly cleaning products.",
            "Switch to rechargeable batteries.",
            "Return expired medicines to pharmacies.",
            "Choose LED bulbs over fluorescent (no mercury).",
        ],
        "hazard_level": "high",
    },
    "e_waste": {
        "category": "E-Waste — Electronic Waste",
        "icon": "📱",
        "color": "#9C27B0",
        "disposal": [
            "Do NOT throw electronics in regular bins.",
            "Take to certified e-waste collection centres.",
            "Many electronics brands offer take-back programmes.",
            "Remove batteries before disposal.",
            "Erase personal data from phones and laptops.",
        ],
        "recycling": [
            "Circuit boards → gold, silver, copper, palladium recovery.",
            "Screens → glass and rare earth element recovery.",
            "Cables → copper wire extraction.",
            "Phones → refurbishment or component harvesting.",
        ],
        "impact": {
            "decomposition_years": "Forever — toxic heavy metals persist",
            "co2_saved_kg": 20.0,
            "water_saved_liters": 100.0,
            "energy_saved_kwh": 50.0,
            "fact": "1 million recycled phones can recover 35,000 lbs of copper and 75 lbs of gold.",
        },
        "tips": [
            "Repair before replacing electronics.",
            "Donate working devices to NGOs.",
            "Use certified e-waste recyclers (check CPCB list in India).",
            "Buy refurbished electronics when possible.",
        ],
        "hazard_level": "medium",
    },
    "glass": {
        "category": "Recyclable — Glass",
        "icon": "🪟",
        "color": "#00BCD4",
        "disposal": [
            "Rinse glass bottles and jars before recycling.",
            "Remove metal caps and lids (recycle separately).",
            "Place in the BLUE recycling bin.",
            "Do NOT mix broken glass with regular waste — wrap safely.",
            "Window glass, mirrors, and ceramics are NOT accepted in bottle recycling.",
        ],
        "recycling": [
            "Glass → new bottles and jars (infinitely recyclable with no quality loss).",
            "Crushed glass (cullet) → used in road construction, insulation, tiles.",
            "Coloured glass must be separated: clear, green, brown.",
        ],
        "impact": {
            "decomposition_years": "1 million years (practically forever)",
            "co2_saved_kg": 0.3,
            "water_saved_liters": 2.0,
            "energy_saved_kwh": 1.2,
            "fact": "Recycling glass reduces energy usage by 30% compared to making new glass from sand.",
        },
        "tips": [
            "Reuse glass jars for storage before recycling.",
            "Glass is infinitely recyclable — it never loses quality.",
            "Avoid putting broken glass loose in the bin; wrap in newspaper.",
        ],
        "hazard_level": "low",
    },
    "medical": {
        "category": "Biomedical Waste — Special Handling Required",
        "icon": "🏥",
        "color": "#E91E63",
        "disposal": [
            "⚠️ NEVER place in household bins!",
            "Sharps (needles, blades): use puncture-proof containers.",
            "Used masks/gloves: wrap in paper, place in yellow biomedical bags.",
            "Expired medicines: return to pharmacies or authorised centres.",
            "Bandages/dressings: yellow bag for biomedical waste.",
        ],
        "recycling": [
            "Most biomedical waste must be incinerated.",
            "Pharmaceutical waste → incineration or chemical treatment.",
            "Some plastics (non-contaminated IV bags) can be autoclaved and recycled.",
        ],
        "impact": {
            "decomposition_years": "Varies — many items are biohazardous regardless",
            "co2_saved_kg": 0.0,
            "water_saved_liters": 0.0,
            "energy_saved_kwh": 0.0,
            "fact": "Improperly discarded needles cause ~2 million needlestick injuries globally each year.",
        },
        "tips": [
            "Never flush medicines down the toilet — they contaminate water.",
            "Use designated disposal bins at hospitals and clinics.",
            "Report illegal dumping of medical waste to local authorities.",
        ],
        "hazard_level": "high",
    },
    "textile": {
        "category": "Textile Waste",
        "icon": "👕",
        "color": "#7E57C2",
        "disposal": [
            "Donate usable clothes to NGOs, shelters, or collection drives.",
            "Old torn textiles → place in dry waste bin.",
            "Shoes: donate usable pairs; bin only unusable ones.",
            "Avoid putting textiles in wet waste — they rot and produce methane.",
        ],
        "recycling": [
            "Cotton → industrial wiping cloths, insulation, paper.",
            "Polyester → shredded and remade into new polyester fibre.",
            "Wool → reprocessed into blankets, insulation.",
            "Mixed/blended fabrics → difficult to recycle (usually downcycled).",
        ],
        "impact": {
            "decomposition_years": "Cotton: 5 months | Polyester: 200+ years | Nylon: 30–40 years",
            "co2_saved_kg": 3.6,
            "water_saved_liters": 7000.0,
            "energy_saved_kwh": 6.0,
            "fact": "The fashion industry produces 10% of global CO₂ — more than aviation and shipping combined.",
        },
        "tips": [
            "Buy fewer, higher-quality clothes.",
            "Try clothing swaps with friends.",
            "Repair clothes instead of throwing them away.",
            "Choose natural fibres (cotton, linen, hemp) over synthetics when possible.",
        ],
        "reuse": [
            "🧹 Dusting Rags: Cut old cotton t-shirts into reusable cleaning cloths.",
            "🛍️ No-Sew Tote Bags: Transform old t-shirts into eco-friendly grocery bags.",
            "🛋️ Cushion Stuffing: Shred fabric scraps to fill pillows or pet beds.",
        ],
        "hazard_level": "low",
    },
}

# Add reuse field to all WASTE_KNOWLEDGE entries
WASTE_KNOWLEDGE["plastic"]["reuse"] = [
    "🪴 Seed Starters & Planters: Cut bottles in half to make self-watering plant pots.",
    "🚿 Garden Sprinkler: Poke small holes in bottle caps for gentle watering.",
    "🖊️ Desk Organiser: Cut off tops of containers/jugs for stationery or tool storage.",
    "🐦 DIY Bird Feeder: Push wooden spoons through a bottle to make a bird feeder.",
]

WASTE_KNOWLEDGE["recyclable"]["reuse"] = [
    "📦 Storage & Shipping: Keep sturdy cardboard boxes for organizing or moving.",
    "🌿 Garden Mulch: Shred unprinted cardboard to create weed barriers in garden beds.",
    "🎨 Crafts & DIY: Use cardboard for drawer dividers, cat scratchers, or toys.",
    "📄 Scrap Notes: Use blank sides of printed paper for notes and lists.",
]

WASTE_KNOWLEDGE["metal"]["reuse"] = [
    "🪴 Herb Planters: Punch drainage holes in tin cans to grow kitchen herbs or succulents.",
    "✏️ Pen Holders: Paint or wrap tin cans in twine to make desk pencil & utensil holders.",
    "🕯️ Candle Lanterns: Punch decorative nail holes in tin cans to create candle lanterns.",
    "🔧 Hardware Bins: Use clean cans to store nails, screws, and workshop items.",
]

WASTE_KNOWLEDGE["dry_waste"]["reuse"] = [
    "🧹 Cleaning Rags: Cut worn-out fabrics/towels into reusable dust cloths.",
    "🎨 Craft Projects: Use scrap rubber, ceramic, or tiles for DIY home decor.",
]

WASTE_KNOWLEDGE["wet_waste"]["reuse"] = [
    "🌱 Kitchen Composting: Turn food scraps into nutrient-rich compost for gardens.",
    "🍌 Plant Fertiliser: Use banana peels and crushed eggshells directly in garden soil.",
    "☕ Soil Booster: Sprinkle dried coffee grounds around plants for nitrogen booster.",
]

WASTE_KNOWLEDGE["hazardous"]["reuse"] = [
    "⚡ Rechargeable Batteries: Switch to rechargeable batteries to reuse 500+ times.",
    "🎨 Storage Buckets: Thoroughly clean non-toxic paint buckets for garden storage.",
]

WASTE_KNOWLEDGE["e_waste"]["reuse"] = [
    "📱 Smart Clock / Security Cam: Convert old smartphones into bedside clocks or security monitors.",
    "💻 Refurbish & Donate: Pass working electronics to schools, NGOs, or relatives.",
]

WASTE_KNOWLEDGE["glass"]["reuse"] = [
    "🫙 Food Storage: Wash jars to store dry beans, spices, leftover soups, or jams.",
    "🕯️ Vases & Candles: Use glass bottles as flower vases or DIY candle holders.",
    "🖊️ Desk Storage: Use small glass jars to organize paperclips, buttons, or coins.",
]

WASTE_KNOWLEDGE["medical"]["reuse"] = [
    "😷 Washable Cloth Masks: Use washable cloth masks instead of disposable ones.",
    "💊 Craft Organisers: Clean empty medicine pill bottles for small craft storage.",
]


# ─── Expanded alias map for smarter matching ─────────────────────────────────────
# Each entry: waste_key → list of keywords/phrases that map to it
ALIAS_MAP = {
    "plastic": [
        "plastic", "plastics", "bottle", "bottles", "pet bottle", "water bottle",
        "plastic bag", "polythene", "polybag", "wrapper", "wrappers", "packaging",
        "pet", "hdpe", "pvc", "ldpe", "pp", "ps", "polystyrene", "styrofoam",
        "thermocol", "cling wrap", "plastic wrap", "single use", "single-use",
        "disposable cup", "plastic cup", "plastic container", "tupperware",
        "microplastic", "microplastics", "plastic pollution", "plastic straw",
        "straw", "sachets", "sachet", "chips packet", "chip bag",
    ],
    "recyclable": [
        "paper", "cardboard", "newspaper", "magazine", "carton",
        "office paper", "printer paper", "notebook", "book",
        "tissue box", "cereal box", "envelope", "junk mail",
        "recyclable", "recycling", "recycle",
    ],
    "glass": [
        "glass", "glass bottle", "glass jar", "broken glass", "mirror",
        "wine bottle", "beer bottle", "tumbler", "glass container",
    ],
    "metal": [
        "metal", "cans", "tin", "tin can", "aluminium", "aluminum",
        "steel", "iron", "copper", "foil", "aluminium foil", "aluminum foil",
        "scrap metal", "scrap", "aerosol", "spray can", "soda can",
        "food can", "beverage can", "metal can", "drink can",
    ],
    "wet_waste": [
        "food", "food waste", "food scrap", "food scraps", "organic",
        "organic waste", "fruit", "fruit peel", "banana peel", "vegetable",
        "kitchen waste", "kitchen scrap", "leftover", "leftovers",
        "tea bag", "coffee grounds", "eggshell", "egg shell",
        "garden waste", "grass", "leaves", "flower", "flowers",
        "cooked food", "rotten food", "expired food",
    ],
    "dry_waste": [
        "dry waste", "dry", "fabric", "rubber", "ceramic",
        "sanitary", "diaper", "diapers", "nappy", "sanitary pad",
        "soiled paper", "tissue", "tissues", "tetra pak", "tetra pack",
        "multilayer", "multi-layer", "chips packet",
    ],
    "hazardous": [
        "hazardous", "chemical", "chemicals", "battery", "batteries",
        "paint", "paints", "medicine", "medicines", "drug", "drugs",
        "acid", "toxic", "poison", "pesticide", "insecticide",
        "bleach", "ammonia", "nail polish", "nail polish remover",
        "thermometer", "mercury", "fluorescent", "cfl",
        "motor oil", "engine oil", "antifreeze", "solvent",
    ],
    "e_waste": [
        "e-waste", "e waste", "ewaste", "electronic", "electronics",
        "phone", "mobile", "smartphone", "laptop", "computer",
        "cable", "cables", "charger", "adapter", "headphone",
        "earphone", "circuit board", "pcb", "motherboard",
        "printer", "keyboard", "mouse", "monitor", "tv",
        "television", "remote", "router", "modem",
        "microwave", "refrigerator", "fridge", "washing machine",
    ],
    "medical": [
        "medical", "biomedical", "syringe", "needle", "needles",
        "bandage", "surgical", "gloves", "mask", "masks",
        "blood", "ppe", "hospital waste", "clinical waste",
        "insulin", "lancet", "catheter", "iv bag",
    ],
    "textile": [
        "textile", "textiles", "cloth", "clothes", "clothing",
        "fabric", "shirt", "pants", "jeans", "dress",
        "shoe", "shoes", "leather", "cotton", "polyester",
        "nylon", "wool", "silk", "denim", "sweater",
        "jacket", "socks", "underwear", "uniform",
    ],
}

# ─── Item-specific knowledge for direct questions ────────────────────────────────

ITEM_KNOWLEDGE = {
    "bottle": {
        "answer": (
            "🧴 **Bottles** — depends on the material:\n\n"
            "**Plastic bottle (PET #1):**\n"
            "  • Rinse, crush, and place in the BLUE recycling bin\n"
            "  • Can be recycled into polyester fibre and new bottles\n"
            "  • Takes 450 years to decompose\n\n"
            "**Glass bottle:**\n"
            "  • Rinse and place in the BLUE recycling bin\n"
            "  • Infinitely recyclable with no quality loss\n"
            "  • Takes 1 million years to decompose naturally\n\n"
            "**Metal/aluminium bottle:**\n"
            "  • Rinse and recycle — takes only 60 days to become a new can\n\n"
            "💡 Always check the recycling symbol on the bottom!"
        ),
    },
    "battery": {
        "answer": (
            "🔋 **Batteries — Special Disposal Required!**\n\n"
            "⚠️ **Never throw batteries in regular bins!**\n\n"
            "**Alkaline batteries (AA, AAA, D):**\n"
            "  • Take to battery drop-off points at electronics stores\n"
            "  • Some municipalities accept in hazardous waste drives\n\n"
            "**Lithium-ion batteries (phone, laptop):**\n"
            "  • Return to manufacturer or certified e-waste centre\n"
            "  • Risk of fire if damaged — store safely before disposal\n\n"
            "**Car/lead-acid batteries:**\n"
            "  • Return to auto parts stores or scrap dealers\n"
            "  • Lead is 99% recyclable\n\n"
            "**Button cell batteries:**\n"
            "  • Contain mercury/silver — take to hazardous waste facility\n\n"
            "🌍 **Fact:** A single battery can contaminate 600,000 litres of water."
        ),
    },
    "straw": {
        "answer": (
            "🥤 **Straws — Plastic Straws are a Major Pollutant!**\n\n"
            "**Plastic straws:**\n"
            "  • Too small to recycle — go to DRY waste / landfill\n"
            "  • Take 200 years to decompose\n"
            "  • Among top 10 items found in ocean cleanup\n\n"
            "**Better alternatives:**\n"
            "  • 🥤 Metal/stainless steel straws (reusable)\n"
            "  • 🎍 Bamboo straws (biodegradable)\n"
            "  • 📄 Paper straws (compostable)\n"
            "  • 🌾 Wheat/rice straws (edible!)\n"
            "  • Or just drink directly from the glass! 😊\n\n"
            "🌍 **Fact:** Over 8.3 billion plastic straws pollute the world's beaches."
        ),
    },
    "mask": {
        "answer": (
            "😷 **Face Masks — Proper Disposal is Critical!**\n\n"
            "**Disposable surgical masks:**\n"
            "  • Cut the ear loops to prevent animal entanglement\n"
            "  • Wrap in paper and place in DRY waste bin\n"
            "  • In hospitals: yellow biomedical waste bag\n"
            "  • Takes 450 years to decompose (polypropylene)\n\n"
            "**N95/KN95 masks:**\n"
            "  • Same as surgical masks — dry waste after use\n"
            "  • Some can be reused 5 times (CDC guidelines)\n\n"
            "**Cloth masks:**\n"
            "  • Wash and reuse! Most eco-friendly option\n"
            "  • When worn out → textile/dry waste bin\n\n"
            "⚠️ **Never flush masks — they clog sewage systems.**"
        ),
    },
    "diaper": {
        "answer": (
            "🧷 **Diapers — Dry Waste (Non-Recyclable)**\n\n"
            "**Disposal steps:**\n"
            "  1. Roll up tightly and secure with the tabs\n"
            "  2. Place in a sealed bag to contain odour\n"
            "  3. Place in the DRY waste bin\n"
            "  4. They go to sanitary landfill\n\n"
            "**Environmental impact:**\n"
            "  • A single disposable diaper takes 500 years to decompose\n"
            "  • A baby uses ~6,000 diapers before toilet training\n\n"
            "**Eco-friendly alternatives:**\n"
            "  • 🌿 Biodegradable diapers (decompose in 3-6 years)\n"
            "  • 👶 Cloth diapers (reusable, wash and repeat)\n"
            "  • 🌱 Compostable inserts with cloth covers"
        ),
    },
}


# ─── FAQ Knowledge ───────────────────────────────────────────────────────────────

FAQ_RESPONSES = {
    "what is waste segregation": (
        "**Waste segregation** is the process of separating waste into distinct categories:\n\n"
        "🔵 **Blue Bin** — Dry recyclable waste (plastic, paper, metal, glass)\n"
        "🟢 **Green Bin** — Wet/organic waste (food scraps, garden waste)\n"
        "🔴 **Red Bin** — Hazardous/e-waste (batteries, chemicals, electronics)\n\n"
        "**Why it matters:**\n"
        "• Makes recycling possible (mixed waste can't be recycled)\n"
        "• Reduces landfill burden by up to 80%\n"
        "• Prevents soil and water contamination from hazardous items\n"
        "• It's the law in many Indian cities under SWM Rules 2016"
    ),
    "why recycle": (
        "**Recycling matters because:**\n\n"
        "♻️ Conserves natural resources (trees, water, minerals)\n"
        "⚡ Saves energy (recycling aluminium uses 95% less energy)\n"
        "🌍 Reduces greenhouse gas emissions\n"
        "🏭 Creates jobs in the recycling industry\n"
        "🌊 Prevents ocean and soil pollution\n\n"
        "In India, proper recycling can divert up to 80% of waste from landfills."
    ),
    "composting": (
        "**Home Composting Guide:**\n\n"
        "1. Get a compost bin (or DIY with a bucket with holes)\n"
        "2. Add 'greens' — food scraps, coffee grounds, grass clippings\n"
        "3. Add 'browns' — dry leaves, newspaper, cardboard\n"
        "4. Keep ratio: 3 parts brown to 1 part green\n"
        "5. Turn weekly, keep moist (not wet)\n"
        "6. Ready in 2-3 months — dark, crumbly, earthy smell\n\n"
        "💡 **Vermicomposting** (with earthworms) is great for apartments!\n"
        "💡 **Bokashi** composting can handle meat and dairy too."
    ),
    "reduce plastic": (
        "**10 Ways to Reduce Plastic Waste:**\n\n"
        "1. 🛍️ Carry reusable shopping bags\n"
        "2. 🥤 Use a reusable water bottle\n"
        "3. 🥡 Bring your own container for takeaway\n"
        "4. 🧴 Buy refillable products\n"
        "5. 🧃 Avoid single-use straws\n"
        "6. 🫧 Use bar soap instead of liquid in plastic bottles\n"
        "7. 📦 Choose products with less packaging\n"
        "8. 🍦 Choose cones over cups\n"
        "9. 🧹 Use a bamboo toothbrush\n"
        "10. 💰 Support plastic-free brands"
    ),
    "swachh bharat": (
        "**Swachh Bharat Mission** guidelines for waste:\n\n"
        "🏠 **Household Level:**\n"
        "• Segregate waste into wet, dry, and hazardous at source\n"
        "• Hand over segregated waste to the municipal collector\n\n"
        "🏙️ **Municipal Level:**\n"
        "• Door-to-door collection of segregated waste\n"
        "• Wet waste → composting/biogas\n"
        "• Dry waste → Material Recovery Facility (MRF)\n"
        "• Hazardous → Authorised treatment facilities\n\n"
        "📱 Report littering via Swachhata App"
    ),
    "decomposition": (
        "**How Long Does Waste Take to Decompose?**\n\n"
        "| Item                | Time               |\n"
        "|---------------------|---------------------|\n"
        "| Banana peel         | 2–5 weeks           |\n"
        "| Paper               | 2–6 weeks           |\n"
        "| Cotton cloth        | 5 months            |\n"
        "| Cardboard           | 2 months            |\n"
        "| Plywood             | 1–3 years           |\n"
        "| Cigarette butt      | 10 years            |\n"
        "| Leather shoe        | 25–40 years         |\n"
        "| Nylon fabric        | 30–40 years         |\n"
        "| Tin/aluminium can   | 50–200 years        |\n"
        "| Plastic bag         | 500 years           |\n"
        "| Plastic bottle      | 450 years           |\n"
        "| Styrofoam           | 500+ years          |\n"
        "| Glass bottle        | 1 million years     |\n"
        "| Diaper              | 500 years           |\n\n"
        "🌍 Proper disposal and recycling can prevent centuries of pollution!"
    ),
    "microplastic": (
        "**Microplastics — The Invisible Threat:**\n\n"
        "Microplastics are tiny plastic fragments (<5mm) found EVERYWHERE:\n\n"
        "📍 **Where they come from:**\n"
        "  • Breakdown of larger plastics (bottles, bags)\n"
        "  • Synthetic clothing fibres (washing polyester releases thousands)\n"
        "  • Tyre dust from driving\n"
        "  • Cosmetics with microbeads\n\n"
        "⚠️ **Why they're dangerous:**\n"
        "  • Found in drinking water, seafood, and even human blood\n"
        "  • Cannot be filtered by most water treatment plants\n"
        "  • Absorb toxic chemicals and enter the food chain\n\n"
        "**What you can do:**\n"
        "  • Use a laundry filter bag for synthetic clothes\n"
        "  • Avoid cosmetics with 'polyethylene' in ingredients\n"
        "  • Reduce plastic use overall"
    ),
    "zero waste": (
        "**Zero Waste Lifestyle — Getting Started:**\n\n"
        "The 5 R's (in order of priority):\n\n"
        "1. **Refuse** — Say no to things you don't need\n"
        "2. **Reduce** — Minimise what you consume\n"
        "3. **Reuse** — Use items multiple times\n"
        "4. **Recycle** — Properly sort recyclables\n"
        "5. **Rot** — Compost organic waste\n\n"
        "**Easy first steps:**\n"
        "• Carry a reusable bag, bottle, and container\n"
        "• Buy in bulk to reduce packaging\n"
        "• Choose products in recyclable/compostable packaging\n"
        "• Start composting kitchen scraps\n"
        "• Repair before replacing"
    ),
}


# ─── Knowledge Encyclopedia & Web Knowledge Extraction ───────────────────────────

KNOWLEDGE_ENCYCLOPEDIA = {
    "burn_plastic": {
        "title": "Burning Plastic & Waste Hazards",
        "keywords": ["burn plastic", "burning plastic", "burn waste", "burning waste", "can i burn", "can we burn", "fire plastic"],
        "simple_explanation": "Burning plastic or garbage at home is extremely dangerous. It releases toxic carcinogenic fumes (dioxins, furans, and black carbon) that poison household air and cause lung illness.",
        "recyclable_status": "❌ Never burn plastic. Almost all plastics (PET bottles, HDPE jugs, PP tubs) can be safely recycled or upcycled into household items instead.",
        "household_advice": "Segregate dry plastics into the Blue Recycling bin or give them to local scrap dealers (kabadiwalas) for mechanical recycling.",
        "eco_impact": "Burning 1 kg of plastic releases up to 3 kg of greenhouse gases and toxic soot into our neighborhoods.",
    },
    "ocean_plastic": {
        "title": "Ocean Plastics & Marine Ecology",
        "keywords": ["ocean plastic", "plastic in ocean", "marine debris", "garbage patch", "sea plastic"],
        "simple_explanation": "Over 10 million tons of plastic enter rivers and oceans each year, forming massive marine garbage gyres and entangling marine wildlife.",
        "recyclable_status": "🌊 Sea creatures often mistake floating plastic bags for jellyfish or ingest tiny floating fragments.",
        "household_advice": "Switch to reusable bags, refuse single-use plastic straws and cups, and always dispose of plastics in closed recycling bins so wind cannot blow them into stormwater drains.",
        "eco_impact": "Plastics take 450 to 1,000 years to break down in seawater, continuously fragmenting into microplastics.",
    },
    "microplastic": {
        "title": "Microplastics — The Invisible Pollution",
        "keywords": ["microplastic", "microplastics", "microbead", "microbeads"],
        "simple_explanation": "Microplastics are tiny plastic specks smaller than 5 mm that shed from synthetic clothing, car tire friction, and degrading plastic litter.",
        "recyclable_status": "⚠️ They cannot be caught by standard water filtration and have been found in tap water, sea salt, honey, and human blood.",
        "household_advice": "Wash synthetic clothes (polyester/nylon) in full laundry loads to reduce fiber shedding, choose natural cotton fabrics, and avoid cosmetic face washes with plastic microbeads.",
        "eco_impact": "Microplastics attract toxic chemicals and are eaten by plankton and fish, moving straight up the human food chain.",
    },
    "styrofoam_thermocol": {
        "title": "Styrofoam & Thermocol (Polystyrene #6)",
        "keywords": ["styrofoam", "thermocol", "polystyrene", "ps plastic", "type 6", "#6 plastic"],
        "simple_explanation": "Styrofoam (expanded polystyrene) is 95% air and 5% plastic resin. It is lightweight, brittle, and breaks apart into millions of crumbs.",
        "recyclable_status": "⚠️ Rarely accepted in municipal blue curbside bins because transporting voluminous air is economically unviable.",
        "household_advice": "Break clean thermocol chunks into plant pot bases for light drainage, or save packaging blocks to cushion fragile items in parcel shipping.",
        "eco_impact": "Styrofoam takes over 500 years to decompose and can leach styrene chemicals if subjected to hot food or drinks.",
    },
    "pizza_box": {
        "title": "Pizza Boxes & Greasy Cardboard",
        "keywords": ["pizza box", "pizza boxes", "greasy box", "dirty cardboard", "greasy cardboard"],
        "simple_explanation": "Pizza boxes are made of sturdy corrugated cardboard, but cooking grease, melted cheese, and tomato sauce soak deeply into the paper pulp.",
        "recyclable_status": "⚠️ Greasy cardboard cannot be pulped into clean new paper because oil ruins the fiber-bonding process.",
        "household_advice": "Tear the box into two: the clean, unsoiled top lid goes straight into Blue Paper Recycling; the oily bottom half goes into Green Wet Waste or home compost!",
        "eco_impact": "Segregating the clean top lid saves thousands of clean cardboard fibers from being unnecessarily dumped into landfills.",
    },
    "tetra_pak": {
        "title": "Tetra Paks & Juice/Milk Cartons",
        "keywords": ["tetra pak", "tetrapak", "juice carton", "milk carton", "aseptic carton"],
        "simple_explanation": "Tetra Paks are composite multi-layer packages made of 75% paperboard, 20% polyethylene film, and 5% ultra-thin aluminum foil.",
        "recyclable_status": "✅ Yes! Specialized paper hydrapulpers separate the high-grade paper fibers from the plastic-aluminum (PolyAl) layers.",
        "household_advice": "Rinse the carton, flatten it completely flat to save space, and place it in your Blue Dry Recycling bin.",
        "eco_impact": "The recovered PolyAl mix is pressed into waterproof building sheets, school desks, and durable panel boards.",
    },
    "blister_pack": {
        "title": "Medicine Blister Packs & Tablet Strips",
        "keywords": ["blister pack", "blister packs", "medicine foil", "pill packet", "tablet strip", "tablet wrapper"],
        "simple_explanation": "Medicine blister packs seal pills between PVC plastic and a thin aluminum foil barrier to keep moisture out.",
        "recyclable_status": "⚠️ Difficult for curbside bins because the bonded plastic and metal layers cannot be peeled apart easily by municipal sorters.",
        "household_advice": "Check if your neighborhood pharmacy or hospital runs a blister pack collection take-back program. If not, dispose of empty strips in Dry Non-Recyclable waste.",
        "eco_impact": "Dedicated recycling programs use chemical pyrolysis to cleanly recover the high-purity aluminum.",
    },
    "batteries": {
        "title": "Batteries & Chemical Safety",
        "keywords": ["battery", "batteries", "aa battery", "lithium", "rechargeable", "button cell", "cell battery"],
        "simple_explanation": "Household batteries contain heavy metals (lithium, lead, cadmium, nickel) and caustic electrolytes.",
        "recyclable_status": "⚠️ NEVER throw batteries into ordinary trash! In garbage trucks or landfills, crushed batteries cause dangerous chemical fires and poison groundwater.",
        "household_advice": "Cover battery terminals with clear tape to prevent short circuits. Drop them off at electronics stores or designated battery collection kiosks.",
        "eco_impact": "Over 95% of materials in modern batteries can be recovered to manufacture new batteries and electronics.",
    },
    "e_waste": {
        "title": "Electronic Waste (E-Waste)",
        "keywords": ["e waste", "electronic waste", "old phone", "laptop", "charger", "cables", "electronics", "old tablet"],
        "simple_explanation": "Old gadgets, dead chargers, laptops, and wires contain valuable precious metals (gold, copper, silver) alongside hazardous lead and mercury.",
        "recyclable_status": "✅ Highly valuable and recyclable when handled by authorized, certified e-waste recyclers.",
        "household_advice": "Store old electronics in a dry cardboard box. Schedule a pickup with certified e-waste recyclers or trade them in during retail electronics promotions.",
        "eco_impact": "Recycling 1 million smartphones recovers roughly 35 kg of gold, 350 kg of silver, and 15 tons of copper!",
    },
    "bubble_wrap": {
        "title": "Bubble Wrap & Air Cushion Packaging",
        "keywords": ["bubble wrap", "air pillows", "packing peanuts", "protective packaging"],
        "simple_explanation": "Bubble wrap is made of Low-Density Polyethylene (#4 LDPE plastic film) inflated with air cushions for shock absorption.",
        "recyclable_status": "✅ Recyclable with flexible polythene films, but loose wrap jams rotating machinery in curbside automated sorting facilities.",
        "household_advice": "Save clean bubble wrap for packing storage, shipping holiday parcels, or winter greenhouse insulation. If discarding, drop off at grocery bag collection bins.",
        "eco_impact": "Reusing packing cushioning prevents new petroleum refining and cuts packaging waste by over 90%.",
    },
    "clothes_textile": {
        "title": "Clothing & Textile Upcycling",
        "keywords": ["clothes", "old clothes", "textiles", "fabric waste", "fast fashion", "bedsheets", "old shirt"],
        "simple_explanation": "Textiles are blends of natural fibers (cotton, wool) and synthetic plastics (polyester, nylon, elastane).",
        "recyclable_status": "✅ Wearable clothes should be donated; unwearable garments can be shredded into industrial insulation, car seat stuffing, or wiping cloths.",
        "household_advice": "Cut worn-out shirts into reusable kitchen dusting rags, turn jeans into sturdy tote bags, or donate gently used clothes to community centers.",
        "eco_impact": "Producing a single cotton t-shirt consumes 2,700 liters of fresh water. Upcycling dramatically protects global water reserves.",
    },
    "broken_glass": {
        "title": "Broken Glass & Worker Safety",
        "keywords": ["broken glass", "shards", "broken mirror", "broken window", "crockery"],
        "simple_explanation": "Broken beverage bottles, jars, window panes, and ceramic shards require special care during handling.",
        "recyclable_status": "Container bottles are 100% infinitely recyclable. Mirrors, window panes, and Pyrex ceramics have different melting points and cannot mix with bottles.",
        "household_advice": "⚠️ Wrap broken glass securely in several sheets of newspaper, tape it firmly, and clearly write 'CAUTION: BROKEN GLASS' before placing into Dry Waste.",
        "eco_impact": "Wrapping protects sanitation and sorting personnel from severe lacerations while ensuring clean glass reaches melting furnaces.",
    },
    "aluminum_foil": {
        "title": "Aluminum Foil & Food Wrappers",
        "keywords": ["aluminum foil", "aluminium foil", "silver foil", "tin foil"],
        "simple_explanation": "Pure metallic aluminum rolled into paper-thin flexible sheets, used for food wrapping and cooking.",
        "recyclable_status": "✅ 100% infinitely recyclable without any loss of metal purity or strength!",
        "household_advice": "Wipe off food sauces. Roll used foil into a ball (at least tennis-ball sized so it doesn't fall through sorting grates) and place in the Blue bin.",
        "eco_impact": "Recycling aluminum consumes 95% less energy than mining and smelting fresh bauxite ore from the earth.",
    },
    "biodegradable_plastic": {
        "title": "Biodegradable & Compostable Plastics (PLA)",
        "keywords": ["biodegradable plastic", "bioplastic", "pla plastic", "polylactic acid", "compostable plastic"],
        "simple_explanation": "Plastics manufactured from plant starch (corn, cassava, sugarcane) engineered to break down under specific heat and moisture conditions.",
        "recyclable_status": "⚠️ Cannot be mixed into standard PET/HDPE recycling bins (it ruins the plastic melt batches), and will NOT break down easily in home gardens.",
        "household_advice": "Send to industrial municipal composting plants capable of reaching 60°C. If unavailable, place in Green Dry Waste.",
        "eco_impact": "Reduces crude oil reliance, but proper segregation is mandatory to avoid contaminating standard recycling streams.",
    },
    "black_plastic": {
        "title": "Black Takeaway Plastic Containers",
        "keywords": ["black plastic", "black container", "black takeaway", "black tub"],
        "simple_explanation": "Takeaway food containers, microwave meal trays, and seedling pots colored with carbon black pigment.",
        "recyclable_status": "⚠️ Automated recycling sorting robots use optical Near-Infrared (NIR) sensors. Carbon black absorbs the light, making items invisible to the cameras!",
        "household_advice": "Wash and reuse black plastic tubs at home for workshop screw storage, snack leftovers, or seed starter trays.",
        "eco_impact": "Opting for clear or light-colored containers ensures automated recycling scanners can detect and sort your plastic.",
    },
    "chips_packet": {
        "title": "Chips Packets & Multilayer Packaging (MLP)",
        "keywords": ["chips packet", "crisp packet", "kurkure", "multilayer packaging", "mlp", "biscuit wrapper"],
        "simple_explanation": "Multilayer packaging combines outer printed plastic with an inner reflective metallized aluminum film to keep food crispy.",
        "recyclable_status": "⚠️ Conventional recyclers cannot separate the bonded plastic and metal layers.",
        "household_advice": "Place in Green Dry Waste. In modern smart cities, MLPs are shredded for co-processing in cement kilns or blended into bitumen roads.",
        "eco_impact": "Try making 'ecobricks' (stuffing dry clean wrappers tightly inside a plastic bottle until rock solid) to build garden borders or furniture!",
    },
    "kabadiwala_informal": {
        "title": "Kabadiwalas & Scrap Collector Networks",
        "keywords": ["kabadiwala", "scrap dealer", "ragpicker", "informal recycling", "waste picker"],
        "simple_explanation": "Local scrap dealers and waste pickers collect, segregate, and channel millions of tons of dry recyclables back into manufacturing.",
        "recyclable_status": "✅ They recover up to 80% of household recyclables (newspapers, cardboard, metal, hard plastics) for direct industrial remanufacturing.",
        "household_advice": "Bundle clean newspapers, cardboard, metal tins, and plastic bottles separately to sell directly to your neighborhood kabadiwala.",
        "eco_impact": "Supports working-class livelihoods, diverts mountains of waste from local dumps, and promotes the circular economy.",
    },
    "cooking_oil": {
        "title": "Used Cooking Oil & Grease Disposal",
        "keywords": ["cooking oil", "used oil", "kitchen grease", "frying oil"],
        "simple_explanation": "Leftover oil and animal fats from deep-frying and kitchen cooking.",
        "recyclable_status": "⚠️ NEVER pour cooking oil into the kitchen sink! Oil hardens inside underground pipes, forming massive 'fatbergs' that clog city sewers.",
        "household_advice": "Let the oil cool. Strain into a sealable glass jar and drop off at biodiesel collection programs, or wipe small pans with paper towels for composting.",
        "eco_impact": "Used cooking oil can be converted into clean-burning biodiesel that reduces vehicle emissions by over 80%.",
    },
    "swachh_bharat": {
        "title": "Swachh Bharat & Waste Segregation",
        "keywords": ["swachh bharat", "swachhata", "clean india", "waste segregation", "segregation rules"],
        "simple_explanation": "The national sanitation guidelines prioritizing 100% waste segregation at the source of generation.",
        "recyclable_status": "Follow the two-bin / three-bin rule: Green Bin for Wet/Organic Waste, Blue Bin for Dry/Recyclables, and Black/Red for Hazardous/Sanitary.",
        "household_advice": "Segregate waste at home right when discarding. Keeping dry recyclables clean and unsoiled protects waste workers and enables recycling.",
        "eco_impact": "Segregation at source boosts recycling efficiency from under 15% to over 85%, dramatically reducing landfill heights.",
    },
}


def fetch_web_knowledge_summary(query: str) -> Optional[Dict]:
    """
    Search Wikipedia REST API for the query and retrieve an accessible summary.
    """
    clean_q = re.sub(r"[^\w\s]", " ", query).strip()
    stop_words = {
        "can", "we", "i", "how", "what", "why", "is", "it", "to", "the",
        "of", "in", "for", "about", "will", "be", "do", "does", "did", "tell", "me"
    }
    tokens = [w for w in clean_q.split() if w.lower() not in stop_words]
    search_term = " ".join(tokens) if tokens else clean_q

    headers = {"User-Agent": "SmartWasteBot/1.0 (waste_advisor@example.com)"}

    try:
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            + urllib.parse.quote(search_term + " recycling waste")
            + "&format=json&utf8=1"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as res:
            data = json.loads(res.read().decode("utf-8"))
            results = data.get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
                sum_req = urllib.request.Request(summary_url, headers=headers)
                with urllib.request.urlopen(sum_req, timeout=3.0) as sres:
                    sdata = json.loads(sres.read().decode("utf-8"))
                    extract = sdata.get("extract", "")
                    if extract:
                        return {"title": sdata.get("title", title), "extract": extract}
    except Exception as e:
        logger.debug(f"Web knowledge search failed for '{query}': {e}")

    return None


try:
    from llama_engine import LlamaEngine
except ImportError:
    LlamaEngine = None

try:
    from executorch_llama_engine import ExecuTorchLlamaEngine
except ImportError:
    ExecuTorchLlamaEngine = None


# ─── Chatbot Engine ──────────────────────────────────────────────────────────────

class WasteAdvisor:
    """
    AI waste advisory chatbot.

    Integrated with Llama 3.2 (1B) running locally via llama-cpp-python
    for the Bharat AI-SoC Challenge PS2 competition.
    Falls back to rule-based responses when model weights are not present.
    """

    def __init__(self, llama_engine=None):
        self.conversation_history: List[Dict] = []
        self.last_detection: Optional[Dict] = None
        self.llama_engine = llama_engine

        # ── Try new LlamaEngine (GGUF-first, universal) ──────────────────
        if self.llama_engine is None and LlamaEngine:
            try:
                from config import DEFAULT_CONFIG
                self.llama_engine = LlamaEngine(config=DEFAULT_CONFIG.llama)
                self.llama_engine.initialize()
                logger.info(f"✅ Llama Engine initialized: {self.llama_engine.load_status}")
            except Exception as e:
                logger.warning(f"⚠️ Llama Engine init error: {e}")
                self.llama_engine = None

        # ── Fallback to legacy ExecuTorch engine ─────────────────────────
        if self.llama_engine is None and ExecuTorchLlamaEngine:
            try:
                self.llama_engine = ExecuTorchLlamaEngine()
                self.llama_engine.initialize()
                logger.info(f"✅ ExecuTorch Llama Engine initialized: {self.llama_engine.load_status}")
            except Exception as e:
                logger.warning(f"⚠️ ExecuTorch Llama Engine init error: {e}")
                self.llama_engine = None

        if self.llama_engine is None:
            logger.info("WasteAdvisor initialized (rule-based mode)")

    def set_last_detection(self, detection: Dict):
        """Store the most recent waste detection for context."""
        self.last_detection = detection

    def get_response(self, user_message: str, detection_context: Optional[Dict] = None) -> Dict:
        """
        Generate a response to a user message.

        Args:
            user_message: The user's question or message
            detection_context: Optional waste detection result for context

        Returns:
            Dict with 'message', 'type', 'waste_info' keys
        """
        if detection_context:
            self.last_detection = detection_context

        message_lower = user_message.strip().lower()
        # Clean up punctuation for matching
        message_clean = re.sub(r'[^\w\s]', ' ', message_lower).strip()
        message_clean = re.sub(r'\s+', ' ', message_clean)

        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        # Route to appropriate handler
        response = self._route_message(message_lower, message_clean, user_message)

        # Store response in history
        self.conversation_history.append({
            "role": "assistant",
            "content": response["message"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        return response

    def guess_object_and_analysis(
        self,
        waste_type: str,
        confidence: float = 0.0,
        bbox: Optional[List[float]] = None,
        detected_object: Optional[str] = None,
        image: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Estimate the specific object identity, material/plastic resin code,
        recycling instructions, and household reuse ideas.
        """
        waste_type_key = (waste_type or "").lower().replace(" ", "_").replace("-", "_")
        obj_key = (detected_object or "").lower().replace("-", " ")

        # Aspect ratio if bbox available: bbox = [x1, y1, x2, y2]
        aspect_ratio = 1.0
        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            w = max(1.0, float(bbox[2]) - float(bbox[0]))
            h = max(1.0, float(bbox[3]) - float(bbox[1]))
            aspect_ratio = h / w

        # Visual crop features if image is provided
        is_metallic = False
        is_cardboard_brown = False
        is_organic_color = False
        if cv2 is not None and np is not None and image is not None and isinstance(image, np.ndarray) and image.size > 0:
            try:
                crop = None
                if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    ih, iw = image.shape[:2]
                    x1 = max(0, min(iw - 1, x1))
                    x2 = max(0, min(iw, x2))
                    y1 = max(0, min(ih - 1, y1))
                    y2 = max(0, min(ih, y2))
                    if x2 > x1 + 8 and y2 > y1 + 8:
                        crop = image[y1:y2, x1:x2]
                if crop is None:
                    ih, iw = image.shape[:2]
                    crop = image[int(ih * 0.15):int(ih * 0.85), int(iw * 0.15):int(iw * 0.85)]

                if crop is not None and crop.size > 0:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    h_chan = hsv[:, :, 0]
                    s_chan = hsv[:, :, 1]
                    v_chan = hsv[:, :, 2]
                    mean_s = float(np.mean(s_chan))
                    mean_v = float(np.mean(v_chan))

                    # Silver/metallic sheen
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if mean_s < 45 and 50 < mean_v < 240 and laplacian_var > 120:
                        is_metallic = True

                    # Cardboard kraft brown
                    brown_mask = (h_chan >= 10) & (h_chan <= 28) & (s_chan >= 35) & (s_chan <= 170) & (v_chan >= 45) & (v_chan <= 210)
                    if float(np.mean(brown_mask)) > 0.40:
                        is_cardboard_brown = True

                    # Organic fruit/veggie color
                    green_mask = (h_chan >= 35) & (h_chan <= 85) & (s_chan >= 60)
                    orange_mask = ((h_chan <= 25) | (h_chan >= 165)) & (s_chan >= 90) & (v_chan >= 70)
                    if float(np.mean(green_mask | orange_mask)) > 0.35:
                        is_organic_color = True
            except Exception as e:
                logger.debug(f"Crop feature analysis error: {e}")

        # Combined signal checks
        has_bottle = ("bottle" in obj_key) or ("bottle" in waste_type_key)
        has_explicit_non_organic = has_bottle or ("plastic" in waste_type_key) or ("metal" in waste_type_key) or ("glass" in waste_type_key) or ("cardboard" in waste_type_key)

        is_spoon_cutlery = any(k in obj_key for k in ["spoon", "fork", "knife", "cutlery"]) or any(k in waste_type_key for k in ["spoon", "cutlery"])
        is_textile = any(k in obj_key for k in ["cloth", "textile", "tie", "umbrella", "backpack", "handbag", "suitcase", "garment", "fabric", "shirt", "towel", "teddy bear"]) or any(k in waste_type_key for k in ["cloth", "textile", "fabric"])
        is_chair_furniture = any(k in obj_key for k in ["chair", "stool", "furniture", "couch", "sofa", "bench", "table", "desk", "seat"]) or any(k in waste_type_key for k in ["chair", "furniture"])
        is_sheet_folder = any(k in obj_key for k in ["sheet", "folder", "sleeve", "film", "cover", "packaging", "acrylic", "clear plastic", "document"]) or any(k in waste_type_key for k in ["sheet", "folder", "film"])
        has_plastic_context = ("plastic" in waste_type_key) or ("plastic" in obj_key) or any(k in obj_key for k in ["jerrycan", "canister", "bottle", "jug", "folder", "sheet", "container", "tub", "vase", "jar"])
        metal_tokens = ["metal", "tin", "scissors", "aluminium", "aluminum", "tinplate", "steel", "metal_can"]
        is_metal_word = any(k in obj_key for k in metal_tokens) or any(w in obj_key.split() for w in ["can", "cans"])
        is_can_metal = (is_metal_word or ("metal" in waste_type_key) or (is_metallic and not has_explicit_non_organic)) and not is_chair_furniture and not is_sheet_folder and not has_plastic_context
        is_organic = any(k in obj_key for k in ["banana", "apple", "orange", "fruit", "vegetable", "carrot", "broccoli", "sandwich", "pizza", "donut", "cake", "food", "scrap", "organic"]) or any(k in waste_type_key for k in ["wet_waste", "organic", "food"]) or (is_organic_color and not has_explicit_non_organic and not is_spoon_cutlery and not is_textile)
        is_cardboard = any(k in obj_key for k in ["cardboard", "carton", "box", "paper", "book"]) or any(k in waste_type_key for k in ["cardboard", "paper"]) or (is_cardboard_brown and not has_explicit_non_organic and not is_sheet_folder)
        is_glass = (any(k in obj_key for k in ["glass", "wine glass"]) or ("glass" in waste_type_key and not is_sheet_folder)) and not has_plastic_context
        is_ewaste = any(k in obj_key for k in ["phone", "cell phone", "laptop", "mouse", "keyboard", "remote", "electronic", "microwave", "oven", "toaster", "refrigerator", "clock", "hair drier", "e_waste", "ewaste", "appliance", "battery"]) or ("hazardous" in waste_type_key and not is_can_metal)
        is_cup = ("cup" in obj_key) or ("tumbler" in obj_key) or ("cup" in waste_type_key)
        is_bag = any(k in obj_key for k in ["bag", "wrapper", "film", "polythene"]) or ("bag" in waste_type_key)

        # ── Profile 1: Cutlery (Spoon, Fork, Knife) ──────────────────────────
        if is_spoon_cutlery:
            guessed_object = "Reusable / Disposable Cutlery (Spoon, Fork, or Knife)"
            plastic_type = "Polypropylene (PP #5), Polystyrene (PS #6), or Stainless Steel"
            resin_code = "RIC #5 (PP) / RIC #6 (PS) / Stainless Steel"
            recycling_steps = [
                "Rinse off food residue and grease with warm water.",
                "Check material: Stainless steel metal cutlery is 100% infinitely recyclable in metal collection.",
                "Rigid plastic cutlery marked #5 (PP) is accepted in clean dry plastics recycling.",
                "Brittle disposable cutlery marked #6 (PS / Styrene) is non-recyclable curbside → place in Green Dry Waste bin.",
                "Whenever possible, switch to washable stainless steel or bamboo cutlery instead of single-use."
            ]
            household_reuse = [
                "🌱 Garden Plant Markers: Write herb and vegetable names with permanent marker on spoon handles and stick into pots.",
                "🎨 Craft & Paint Stirrers: Use plastic spoons to mix paint pots, stir resin/epoxy, or sculpt clay models.",
                "🥄 Pantry Scoops: Keep inside jars of detergent, baking soda, bird seed, or protein powder as convenient scoops.",
                "🪺 DIY Decorative Lotus Candle Holder: Glue clean spoon heads in concentric circles around a tealight candle.",
                "🔨 Mini Succulent Trowel: Use sturdy spoons as gentle miniature trowels for repotting small cacti and succulents."
            ]
            decomp = "20–30 years (PP) / 500+ years (PS) / 100+ years (Steel)"

        # ── Profile 2: Textiles & Clothing / Fabric ──────────────────────────
        elif is_textile:
            guessed_object = "Used Clothing, Garment, or Fabric Textile Waste"
            plastic_type = "Textile Fibre (Cotton, Polyester, Nylon, or Blended Fabric)"
            resin_code = "PET/PES (Polyester) / Natural Cellulose (Cotton)"
            recycling_steps = [
                "Ensure garments are clean and completely dry (damp clothes mildew and ruin entire collection batches).",
                "If wearable and clean: Donate to local charity, orphanages, or community clothing swap drives.",
                "If worn out or torn: Deposit at apparel store textile collection bins or dedicated fabric recycling hubs.",
                "Do not put fabrics in curbside sorting bins (textiles wrap around and jam sorting facility conveyor rollers)."
            ]
            household_reuse = [
                "🧹 Lint-Free Dusting & Cleaning Rags: Cut worn cotton T-shirts or towels into reusable kitchen and car-cleaning cloths.",
                "🛋️ Cushion & Dog Bed Stuffing: Shred old shirts, socks, and fabric scraps into soft stuffing for throw pillows or pet beds.",
                "🛍️ DIY No-Sew Tote Bag: Cut the sleeves and neckline off an old T-shirt, knot the bottom hem, and make a sturdy grocery bag.",
                "🪴 Hanging Macrame Plant Holder: Braid fabric strips from old clothes into bohemian hanging plant holders.",
                "🧦 Draft Stopper: Stuff a long sleeve or leg from old jeans with scraps and place against doors to block winter drafts."
            ]
            decomp = "5 months (Cotton) / 20–200 years (Synthetic Polyester)"

        # ── Profile 2B: Molded Plastic Chairs, Stools & Rigid Furniture ──────
        elif is_chair_furniture:
            guessed_object = "Molded Plastic Chair, Stool, or Rigid Plastic Furniture"
            plastic_type = "Rigid Polypropylene (PP #5) or High-Density Polyethylene (HDPE #2)"
            resin_code = "RIC #5 (PP) / RIC #2 (HDPE) — Heavy-Duty Molded Plastic"
            recycling_steps = [
                "Do NOT put plastic chairs in standard curbside household bins (their bulky volume jams automated bin trucks).",
                "If intact or repairable: Donate to community centres, schools, event spaces, charities, or thrift shops.",
                "Separate fasteners: Unscrew steel screws, bolts, swivel mounts, or rubber leg feet before recycling.",
                "Drop off at a municipal Bulky Waste / Civic Amenity Centre or commercial industrial plastics granulator.",
                "Molded PP/HDPE chairs are high-value engineering resins: Recyclers shred and pelletize them into shipping pallets, crates, and garden benches."
            ]
            household_reuse = [
                "🪴 Heavy-Duty Garden Plant Stand: Turn an aged or cracked plastic chair into an outdoor riser for heavy potted plants and trailing vines.",
                "🎨 Kids Craft & Painting Station: Designate as a dedicated splatter-proof seat for messy painting, clay modeling, or DIY crafts.",
                "🛠️ Workshop & Garage Utility Seat: Keep in your garage as a durable, grease-resistant stool for car maintenance and workbench tasks.",
                "🐕 Elevated Pet Feeding Station: Cut two round holes in the seat pan to hold pet food and water bowls at an ergonomic height.",
                "🪑 Easy Patio Restoration: Clean weathered plastic using warm water, baking soda, and a dash of dish soap to bring back original shine.",
                "📦 Inverted Storage Caddy: If legs are broken, detach the seat shell and flip upside-down to hold garden hoses, cords, or workshop scrap."
            ]
            decomp = "500+ years (Thick-walled molded PP/HDPE with UV-stabilizers takes centuries to degrade in landfills)"

        # ── Profile 2C: Plastic Sheets, File Folders & Clear Plastic Packaging ─
        elif is_sheet_folder:
            guessed_object = "Plastic Sheet, File Folder, or Clear Plastic Packaging"
            plastic_type = "Polypropylene (PP #5), Rigid PVC #3, or Acrylic"
            resin_code = "RIC #5 (PP) / RIC #3 (PVC)"
            recycling_steps = [
                "Peel off external paper labels, stickers, or adhesive tape.",
                "Ensure clean and completely dry; remove any metal binder rings, paper clips, or staples.",
                "Clean rigid PP plastic folders are 100% recyclable in the BLUE dry recycling bin.",
                "Note: Highly flexible vinyl/PVC sheets should be checked for local soft-plastic drop-off."
            ]
            household_reuse = [
                "📁 Reusable Document & Receipt Organizer: Group home bills, warranties, tax receipts, or medical records.",
                "🎨 DIY Craft Stencil & Cutting Mat Shield: Cut durable stencils for art painting, tracing, or desktop craft protection.",
                "🌱 Seedling Miniature Greenhouse Cover: Drape over potted plant trays to maintain warm humidity during seed germination.",
                "📦 Shelf & Drawer Liner: Line pantry shelves or refrigerator crisper drawers to catch spills for easy wipe-down cleaning.",
                "🔖 DIY Reusable Bookmarks & Dividers: Cut into colorful study bookmarks, recipe dividers, or tabbed binder dividers."
            ]
            decomp = "450–500 years (Dense polymer sheets resist UV degradation)"

        # ── Profile 3: Metal Cans / Tins / Aluminium ──────────────────────────
        elif is_can_metal:
            guessed_object = "Aluminium Beverage Can or Steel Tin Food Can"
            plastic_type = "Non-Plastic — Metal (Aluminium Alloy 3104 / Tinplate Steel)"
            resin_code = "ALU 41 (Aluminium) / FE 40 (Steel)"
            recycling_steps = [
                "Rinse beverage or food residue with water to deter insects and unpleasant odors.",
                "Leave pull-tabs attached or push inside the can.",
                "Crush slightly if needed to save space in your household dry bin.",
                "Place in the BLUE dry recycling bin (infinitely recyclable in 60 days with 95% energy savings!)."
            ]
            household_reuse = [
                "🪴 Kitchen Herb Planter: Punch 2-3 drainage holes in the bottom, paint the exterior, and grow fresh basil, mint, or thyme.",
                "✏️ Desk Pen & Utensil Holder: Smooth the top rim, wrap in jute twine or colorful paint for a chic stationery caddy.",
                "🕯️ Candle Lantern: Fill can with water, freeze solid, punch decorative punched-hole patterns with a nail, and insert a tealight.",
                "🔧 Workshop Hardware Bin: Store screws, washers, drill bits, and craft beads organized on workshop shelves.",
                "🍪 Biscuit / Cookie Cutter: Cut strips from clean aluminium cans and bend into custom baking cookie shapes."
            ]
            decomp = "50–200 years (Steel) / 200–500 years (Aluminium)"

        # ── Profile 4: Organic / Wet Food & Fruit Waste ───────────────────────
        elif is_organic:
            guessed_object = "Organic Food Scrap, Fruit Peel, or Vegetable Waste"
            plastic_type = "100% Biodegradable Biomass (Cellulose, Pectin & Moisture)"
            resin_code = "Organic / Wet Waste"
            recycling_steps = [
                "Separate completely from plastic wrappers, fruit stickers, rubber bands, and twist ties.",
                "Collect in a ventilated green kitchen caddy lined with newspaper or certified compostable bags.",
                "Transfer to home compost bin, vermicompost unit, or your municipal Green Wet Waste collection.",
                "Never put food scraps in the dry recyclables bin (moisture and oils ruin recyclable paper/cardboard bales)."
            ]
            household_reuse = [
                "🌱 Nutrient-Rich Garden Compost: Layer with dry leaves and soil in a backyard bin to create black gold organic fertilizer.",
                "🍌 Banana Peel Plant Booster: Soak banana peels in a jar of water for 48 hours to make potassium-rich water for flowering plants.",
                "🍊 Natural Citrus Kitchen Cleaner: Soak orange or lemon peels in white vinegar for 2 weeks to make a fresh degreasing spray.",
                "🍲 Homemade Vegetable Broth: Freeze carrot tops, onion skins, and celery ends; boil later to make rich homemade broth.",
                "☕ Coffee Grounds & Scrap Pest Barrier: Scatter dried grounds or citrus peels around garden beds to deter snails and ants."
            ]
            decomp = "2–4 weeks (Fruit peels) / 1–3 months in compost"

        # ── Profile 5: Cardboard & Paper Packaging ────────────────────────────
        elif is_cardboard:
            guessed_object = "Cardboard Box, Shipping Carton, or Paper Packaging"
            plastic_type = "Non-Plastic — Cellulose Fibre (Corrugated Cardboard / Kraft Paper)"
            resin_code = "PAP 20 (Corrugated) / PAP 21 (Non-corrugated) / PAP 22 (Paper)"
            recycling_steps = [
                "Peel off plastic shipping tape, labels, and styrofoam/bubble inserts.",
                "Flatten boxes completely to maximize space in your dry bin and collection truck.",
                "Keep completely clean and dry — wet or grease-stained cardboard (like greasy pizza box bottoms) cannot be recycled.",
                "Place in the BLUE dry recycling bin."
            ]
            household_reuse = [
                "📦 Drawer & Shelf Dividers: Cut sturdy cardboard strips to neatly organize socks, ties, underwear, or pantry spices.",
                "🌿 Garden Sheet Mulching (Lasagna Gardening): Lay flat unprinted cardboard over garden soil to suppress weeds naturally.",
                "🐈 Cat Scratcher & Play Castle: Cut strips and glue into a tight scratch pad or build fun castle tunnels for pets.",
                "🎨 Painting Mat & Floor Shield: Lay down large boxes when painting walls, doing DIY crafts, or staining furniture.",
                "📦 Safe Moving & Storage: Reuse undamaged cartons for packing books, seasonal clothes, or holiday decorations."
            ]
            decomp = "2–6 weeks (Paper) / 2 months (Cardboard)"

        # ── Profile 6: Glassware & Glass Bottles / Jars ───────────────────────
        elif is_glass:
            guessed_object = "Glass Beverage Bottle, Food Jar, or Glassware"
            plastic_type = "Non-Plastic — Soda-Lime Silicate Glass"
            resin_code = "GL 70 (Clear) / GL 71 (Green) / GL 72 (Brown)"
            recycling_steps = [
                "Empty all contents and rinse lightly with warm water.",
                "Remove metal lids or plastic caps (recycle caps separately in their respective bins).",
                "Labels do not need to be scraped off — high-temperature glass melting furnaces burn them away cleanly.",
                "Place in the BLUE dry recycling bin (or dedicated glass igloo)."
            ]
            household_reuse = [
                "🫙 Pantry Airtight Storage: Store pulses, rice, spices, cookies, homemade jams, pickles, or overnight oats.",
                "🌱 Hydroponic Propagation Vase: Fill with water and place money plant or herb cuttings on your sunny windowsill.",
                "🕯️ Cozy Candle Lantern / Fairy Light Jar: Place a pillar candle or coil a string of warm fairy lights inside for ambient glow.",
                "🪙 Coin, Button & Stationery Jar: Organize coins, safety pins, hairpins, and desktop stationery.",
                "🥗 Salad-in-a-Jar Lunch: Layer dressing at the bottom, greens at the top for crisp meal-prep lunches."
            ]
            decomp = "1 million years (Practically indestructible in landfills, but 100% infinitely recyclable)"

        # ── Profile 7: E-Waste / Electronics & Appliances ─────────────────────
        elif is_ewaste:
            guessed_object = "Electronic Gadget, Circuit Device, or E-Waste Accessory"
            plastic_type = "Composite — ABS Plastic Housing, PCB Board, Copper, Lithium/Metals"
            resin_code = "WEEE / E-Waste (Hazardous & Precious Metal Recovery)"
            recycling_steps = [
                "NEVER throw electronics or batteries in regular garbage or curbside bins (fire and toxic chemical hazard!).",
                "Wipe personal data / perform factory reset before recycling smart devices.",
                "Remove detachable batteries and tape battery terminals with clear tape to prevent short circuits.",
                "Drop off at authorized E-Waste collection centres, electronics store collection bins, or municipal e-waste drives."
            ]
            household_reuse = [
                "📱 Dedicated Alarm Clock or Media Player: Turn an old smartphone into a nightstand alarm clock or Spotify jukebox.",
                "📹 Home Security Cam: Install apps like AlfredCam to repurpose an old phone into an indoor security monitor.",
                "📚 Dedicated E-Reader: Remove distracting apps and keep an old device strictly for reading Kindle/PDF books.",
                "🔌 Spare Parts & Cables: Keep working USB cables and power bricks in a labeled emergency electronics drawer."
            ]
            decomp = "1,000+ years (Leaches lead, mercury & cadmium into groundwater if landfilled)"

        # ── Profile 8: Cups & Tumblers ────────────────────────────────────────
        elif is_cup:
            guessed_object = "Single-Use Drinking Cup (Paper Coffee Cup or Plastic Cold Cup)"
            plastic_type = "Polypropylene (PP #5), PET #1, or PE-Lined Paperboard"
            resin_code = "RIC #5 (PP) / C/PAP 81 (PE-coated Paper)"
            recycling_steps = [
                "Empty remaining coffee, tea, or ice completely.",
                "Separate plastic lid (PP #5) and cardboard sleeve (PAP 20) — both are recyclable in Blue bin.",
                "Note: Traditional paper coffee cups have an internal plastic polyethylene coating; verify if your city accepts poly-coated paper.",
                "Clear plastic cups (PET #1 or PP #5): Rinse and place in BLUE dry recycling bin."
            ]
            household_reuse = [
                "🪴 Seedling Nursery Pots: Poke drainage holes in the bottom and fill with soil to sprout tomato, chili, or herb seeds.",
                "🎨 Kids Craft Paint Pots: Ideal for holding acrylic paint, rinsing brushes, or sorting beads during craft time.",
                "🧶 Yarn Feeder Cup: Cut a small hole in the lid, put a ball of yarn inside, and pull yarn through without tangling.",
                "🧰 Small Hardware Catch-All: Hold nuts, bolts, or small screws while taking apart furniture or appliances."
            ]
            decomp = "20–30 years (Plastic cup) / 20–30 years (PE-lined cup due to plastic film)"

        # ── Profile 9: Plastic Bags & Flexible Wrappers ───────────────────────
        elif is_bag or (aspect_ratio < 0.65 and "plastic" in waste_type_key):
            guessed_object = "Plastic Bag, Polythene Film, or Food Wrapper"
            plastic_type = "LDPE (RIC #4) or Multilayer Flexible Film"
            resin_code = "RIC #4 (LDPE) / BOPP"
            recycling_steps = [
                "If shiny/metallic inside (chips packet), it is multilayer non-recyclable → Green Dry Waste.",
                "If stretchable clean polythene bag → bundle inside another bag and take to supermarket plastic drop-off.",
                "Never put loose plastic bags in curbside sorting bins (they jam sorting gears)."
            ]
            household_reuse = [
                "🗑️ Small Trash Can Liner: Line bathroom or bedroom wastebaskets.",
                "📦 Parcel Cushioning: Reuse clean plastic bags instead of bubble wrap for shipping packages.",
                "👟 Travel Shoe Protector: Wrap dirty shoes before placing into luggage bags.",
                "🌧️ Waterproof Plant Saucer: Place under plant pots to protect wooden tables from moisture."
            ]
            decomp = "500 years"

        # ── Profile 10: Plastic Containers, Jerrycans & Jugs ──────────────────
        elif any(k in obj_key for k in ["container", "tub", "jug", "jerrycan", "canister", "bowl"]) or (bbox is not None and 0.7 <= aspect_ratio <= 1.3 and "plastic" in waste_type_key and "bottle" not in obj_key and "bottle" not in waste_type_key):
            guessed_object = "Plastic Jerrycan, Heavy Container, or Milk Jug"
            plastic_type = "High-Density Polyethylene (HDPE #2) or Polypropylene (PP #5)"
            resin_code = "RIC #2 (HDPE) / RIC #5 (PP)"
            recycling_steps = [
                "Scrape or pour out all liquid, grease, or detergent residue.",
                "Quick rinse with warm soapy water (residue contaminates plastic recycling batches).",
                "Keep plastic screw caps attached or place them inside if accepted by your municipality.",
                "Place in the BLUE dry recycling bin (HDPE containers are among the most demanded and recyclable plastics!)."
            ]
            household_reuse = [
                "🧰 Hardware & Workshop Bins: Cut the side open to make a sturdy handle-equipped scoop or nail/screw tote.",
                "🪴 Garden Watering Jug: Drill small holes in the cap to create an ergonomic watering jug for house plants.",
                "🧼 Bulk Liquid Dispenser: Wash thoroughly and refill with bulk laundry detergent, liquid soap, or car wash fluid.",
                "🎨 Paint & Brush Water Cups: Sturdy, shatterproof cups for DIY painting and garage projects."
            ]
            decomp = "20–30 years (PP) / 500 years (Heavy-walled HDPE)"

        # ── Profile 11: Plastic Bottles (Tall aspect ratio > 1.3 or explicit bottle) ───
        elif "bottle" in obj_key or "bottle" in waste_type_key or "pet" in obj_key or ("plastic" in waste_type_key and (aspect_ratio > 1.3 or bbox is None)):
            guessed_object = "Single-Use Plastic Bottle (Water / Soft Drink / Beverage)"
            plastic_type = "PET / PETE (Resin Identification Code #1)"
            resin_code = "RIC #1 (PET)"
            recycling_steps = [
                "Empty all liquid and give it a quick rinse with clean water.",
                "Peel off external plastic shrink sleeves/labels if loose.",
                "Crush the bottle body flat to save volume in your blue recycling bin.",
                "Keep the cap screwed on (caps are PP #5; modern recycling facilities sort & process both).",
                "Place in the BLUE dry recycling bin."
            ]
            household_reuse = [
                "🪴 Self-Watering Planter: Cut the bottle in half, invert top half into base with a cotton wick to grow kitchen herbs or mint.",
                "🚿 DIY Garden Sprinkler: Poke needle holes in the cap for a gentle indoor plant watering shower.",
                "🐦 Window Bird Feeder: Push two wooden spoons horizontally through holes as perches, fill with bird seeds, and hang outside.",
                "🖊️ Desk Pen & Tool Organizer: Cut off the top cylinder and wrap with twine or ribbon for desk stationery storage.",
                "🍳 Clean Egg Yolk Separator: Squeeze the bottle slightly over a cracked egg to vacuum-lift and separate the yolk cleanly!"
            ]
            decomp = "450 years"

        # ── Profile 12: General / Other Fallback ──────────────────────────────
        else:
            obj_name = detected_object or waste_type.replace('_', ' ').title()
            guessed_object = f"Detected {obj_name} Item"
            plastic_type = "General Recyclable or Residual Material"
            resin_code = "General"
            recycling_steps = [
                "Check material composition (clean plastic, dry paper, or metal).",
                "Segregate clean dry recyclable portions into the BLUE recycling bin.",
                "Place non-recyclable or soiled residue into the Green Dry Waste bin."
            ]
            household_reuse = [
                "🧹 Cleaning Rags: Cut worn fabrics or towels into reusable dusting cloths.",
                "🎨 DIY Craft Supplies: Repurpose sturdy containers, jars, or hardware for home projects."
            ]
            decomp = "Varies by material"

        return {
            "guessed_object": guessed_object,
            "plastic_type": plastic_type,
            "resin_code": resin_code,
            "recycling_steps": recycling_steps,
            "household_reuse": household_reuse,
            "decomposition_time": decomp,
        }

    def get_detection_advice(
        self,
        waste_type: str,
        confidence: float = 0.0,
        bbox: Optional[List[float]] = None,
        detected_object: Optional[str] = None,
        image: Optional[np.ndarray] = None,
    ) -> Dict:
        """Generate comprehensive disposal, object guess, recycling, and household reuse advice."""
        waste_type_key = waste_type.lower().replace(" ", "_").replace("-", "_")

        analysis = self.guess_object_and_analysis(
            waste_type,
            confidence,
            bbox,
            detected_object=detected_object,
            image=image,
        )

        info = WASTE_KNOWLEDGE.get(waste_type_key)
        if not info:
            for key in WASTE_KNOWLEDGE:
                if key in waste_type_key or waste_type_key in key:
                    info = WASTE_KNOWLEDGE[key]
                    waste_type_key = key
                    break

        if not info:
            info = WASTE_KNOWLEDGE["plastic"]

        icon = info["icon"]
        category = info["category"]
        impact = info["impact"]

        recycling_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(analysis["recycling_steps"]))
        reuse_text = "\n".join(f"  • {h}" for h in analysis["household_reuse"][:4])

        message = (
            f"{icon} **Detected:** {category} (Confidence: {confidence:.0%})\n\n"
            f"🎯 **Object Estimation:** {analysis['guessed_object']}\n"
            f"🏷️ **Likely Material / Resin:** {analysis['plastic_type']}\n\n"
            f"**♻️ How to Recycle This Object:**\n{recycling_text}\n\n"
            f"**🏡 How to Reuse for Household Purpose (DIY Upcycling):**\n{reuse_text}\n\n"
            f"**⏳ Decomposition:** {analysis['decomposition_time']} | 💡 {impact['fact']}"
        )

        return {
            "message": message,
            "type": "detection_advice",
            "waste_info": {
                "waste_type": waste_type_key,
                "category": category,
                "hazard_level": info["hazard_level"],
                "color": info["color"],
                "object_guess": analysis["guessed_object"],
                "plastic_type": analysis["plastic_type"],
                "recycling_steps": analysis["recycling_steps"],
                "household_reuse": analysis["household_reuse"],
            },
        }

    def _format_objects_list_for_detection(self) -> Dict:
        """Format a clear breakdown of all objects captured in the uploaded picture."""
        det_ctx = self.last_detection or {}
        all_dets = det_ctx.get("all_detections") or []
        num_dets = det_ctx.get("num_detections") or len(all_dets) or 1
        top_obj = det_ctx.get("object_guess") or det_ctx.get("detected_object") or "Detected Item"
        waste_type = det_ctx.get("waste_type", "unknown")
        plastic_type = det_ctx.get("plastic_type", "")

        lines = [
            f"📸 **Objects Captured in Your Uploaded Picture ({num_dets} Detected):**\n"
        ]

        if all_dets:
            for idx, item in enumerate(all_dets, 1):
                name = item.get("object") or item.get("detected_object") or item.get("class_name", "Object")
                conf = item.get("confidence", 0.0)
                conf_pct = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else ""
                cat = item.get("class_name", "").replace("_", " ").title()
                lines.append(f"{idx}. **{name}** — Category: *{cat}* ({conf_pct} confidence)")
        else:
            lines.append(f"1. **{top_obj}** — Category: *{waste_type.replace('_', ' ').title()}*")

        lines.append(f"\n💡 **Primary Item Analysis:**")
        lines.append(f"• **Identified As:** {top_obj}")
        if plastic_type:
            lines.append(f"• **Material / Code:** {plastic_type}")

        # Add household reuse summary
        reuse = det_ctx.get("household_reuse") or []
        if reuse:
            lines.append(f"\n🏠 **Top Household Reuse Ideas for this Item:**")
            for r in reuse[:3]:
                lines.append(f"• {r}")

        # Add recycling advice
        steps = det_ctx.get("recycling_steps") or []
        if steps:
            lines.append(f"\n♻️ **Recycling Steps:**")
            for s in steps[:2]:
                lines.append(f"• {s}")

        lines.append("\n_Ask me more about any specific item above, or how to reuse or recycle it!_")

        return {
            "message": "\n".join(lines),
            "type": "detection_objects_list",
            "waste_info": {"waste_type": waste_type, "count": num_dets},
        }

    # ─── Core Routing ────────────────────────────────────────────────────────────

    def _route_message(self, message_lower: str, message_clean: str, original: str) -> Dict:
        """Route user message to the appropriate handler using multi-pass matching."""

        # ── Pass 0: Greetings ─────────────────────────────────────────────────
        greetings = ["hello", "hi", "hey", "good morning", "good evening",
                     "namaste", "howdy", "hola", "greetings"]
        if any(message_clean == g or message_clean.startswith(g + " ") for g in greetings):
            return {
                "message": (
                    "👋 Hello! I'm your **AI Waste Disposal Assistant**.\n\n"
                    "I can help you with:\n"
                    "• 🗑️ **How to dispose** any waste — *'How to dispose of plastic bottles?'*\n"
                    "• ♻️ **Recycling info** — *'Can glass be recycled?'*\n"
                    "• 🌍 **Environmental impact** — *'How long does plastic take to decompose?'*\n"
                    "• 🛢️ **Plastic types** — *'What is PET plastic?'*\n"
                    "• 📚 **Waste education** — *'What is zero waste?'*\n"
                    "• 📷 **Identify waste** — upload a photo in the Detect tab\n\n"
                    "Try the quick buttons below or type any question!"
                ),
                "type": "greeting",
                "waste_info": None,
            }

        # ── Pass 0.5: Help command ────────────────────────────────────────────
        if message_clean in ("help", "what can you do", "commands", "menu", "options"):
            return self._help_response()

        # ── Pass 0.5: Thank you ───────────────────────────────────────────────
        if any(kw in message_clean for kw in ["thank", "thanks", "thx", "appreciate"]):
            return {
                "message": (
                    "😊 You're welcome! Happy to help with waste management.\n\n"
                    "Feel free to ask me anything else about:\n"
                    "• Disposal, recycling, or composting\n"
                    "• Plastic types and decomposition\n"
                    "• Environmental impact of waste\n"
                    "• Or upload a waste image in the Detect tab!"
                ),
                "type": "thanks",
                "waste_info": None,
            }

        # ── Pass 1: Detection context & pronoun resolution ────────────────────
        has_detection = bool(self.last_detection and self.last_detection.get("waste_type"))

        # Explicit keywords signaling detection context
        explicit_detection_keywords = [
            "uploaded", "upload", "i uploaded", "the item i uploaded", "item i uploaded",
            "detected", "detection", "my photo", "my image", "my picture",
            "this item", "this image", "this photo", "what did i upload", "what is this",
            "how to reuse this", "how can this be reused", "how can the item be reused",
            "how the item can be reused", "how to dispose this", "last detection",
            "what did you find", "what was that", "uploaded item", "the item",
            "the object", "this object", "object it be", "guess what object", "guess the object",
            "what object", "guess object", "household purpose", "household reuse",
            "for household", "home reuse", "reuse for household", "recycle how",
            "how the object can be recycle", "how the object can be recycled", "how to recycle",
            "this material", "this plastic", "this waste", "scanned item",
            "captured in picture", "in picture", "in the picture", "in photo", "in the photo",
            "in image", "in the image", "objects captured", "captured", "what are the three objects",
            "three objects", "3 objects", "two objects", "2 objects", "what objects",
            "what are the objects", "how many objects", "what is in the picture", "what is in the image",
            "tell me about the picture", "picture i used", "the picture i used", "picture i uploaded",
            "the picture i uploaded",
        ]

        # Action & attribute intents
        is_objects_list_query = any(kw in message_lower for kw in [
            "what are the objects", "what objects", "three objects", "3 objects", "two objects", "2 objects",
            "objects that are captured", "objects captured", "captured in picture", "captured in the picture",
            "captured in photo", "captured in image", "in the picture", "in picture", "in the photo", "in photo",
            "in the image", "in image", "what did you detect", "what was detected", "what did you find in",
            "what items are in", "what are the items", "how many objects", "how many items",
            "what is in the picture", "what is in the image", "what is in the photo", "describe the picture",
            "describe picture", "tell me about the picture", "tell me about the image", "show detections",
            "list of objects", "all objects"
        ])

        is_reuse_query = any(kw in message_lower for kw in [
            "reusable", "reuse", "reused", "repurpose", "repurposing",
            "another purpose", "other purpose", "another use", "other uses", "other purpose",
            "different purpose", "use for another", "use for other", "use it for",
            "use again", "use it again", "can we use", "can it be used",
            "upcycle", "upcycling", "second life", "household purpose", "household reuse",
            "diy", "craft", "crafts", "what can i do with", "what to do with", "make from it"
        ])

        is_recycle_query = any(kw in message_lower for kw in [
            "recycle", "recyclable", "recycling", "blue bin", "can it be recycled",
            "is it recyclable", "how to recycle", "can i recycle", "recycled into"
        ])

        is_guess_query = any(kw in message_lower for kw in [
            "guess", "what object", "object it be", "identify object", "what is the object",
            "what kind of object", "guess what", "plastic type", "material type", "which plastic"
        ])

        is_decompose_query = any(kw in message_lower for kw in [
            "decompose", "decomposition", "how long", "break down", "breakdown",
            "biodegradable", "degrade", "how many years", "decay", "rot away"
        ])

        is_disposal_query = any(kw in message_lower for kw in [
            "dispose", "throw", "bin", "discard", "where do i put", "which bin",
            "how to throw", "dump"
        ])

        conflicting_waste_key = self._find_waste_key(message_clean)
        # If user explicitly asks about another material (e.g. "what about glass?"), don't force detection context
        is_asking_other_material = bool(conflicting_waste_key and has_detection and conflicting_waste_key != self.last_detection.get("waste_type"))

        is_context_relevant = False
        if any(kw in message_lower for kw in explicit_detection_keywords):
            is_context_relevant = True
        elif has_detection and not is_asking_other_material:
            words = set(re.findall(r'\b\w+\b', message_lower))
            has_pronoun = bool(words.intersection({"it", "this", "that", "item", "object"}))
            if has_pronoun or is_reuse_query or is_recycle_query or is_guess_query or is_decompose_query or is_disposal_query or is_objects_list_query:
                is_context_relevant = True

        if is_context_relevant:
            if has_detection:
                waste_type = self.last_detection.get("waste_type")
                confidence = self.last_detection.get("confidence", 0.0)
                bbox = self.last_detection.get("bbox")
                detected_object = self.last_detection.get("detected_object") or self.last_detection.get("object_guess")

                if is_objects_list_query:
                    return self._format_objects_list_for_detection()
                # Compound multi-part questions (e.g. guess object + recycle + household reuse)
                elif (is_reuse_query and is_recycle_query) or (is_guess_query and (is_reuse_query or is_recycle_query)):
                    return self._format_complete_analysis_for_detection(waste_type, confidence, bbox, detected_object=detected_object)
                elif is_reuse_query:
                    return self._format_household_reuse_for_detection(waste_type, confidence, bbox, detected_object=detected_object)
                elif is_recycle_query:
                    return self._format_recycling_for_detection(waste_type, confidence, bbox, detected_object=detected_object)
                elif is_guess_query:
                    return self._format_object_guess_for_detection(waste_type, confidence, bbox, detected_object=detected_object)
                elif is_decompose_query:
                    return self._format_impact_for_detection(waste_type, confidence)
                elif is_disposal_query:
                    return self.get_detection_advice(waste_type, confidence, bbox, detected_object=detected_object)
                else:
                    return self.get_detection_advice(waste_type, confidence, bbox, detected_object=detected_object)
            else:
                return {
                    "message": (
                        "📷 **No waste item has been uploaded or detected yet!**\n\n"
                        "To analyze and guess the object:\n"
                        "1. Go to the **Scanner** on the left\n"
                        "2. 📤 **Upload an image** or 📹 **Start the webcam** to scan your item\n\n"
                        "Once detected, I will guess the specific object, show you how to recycle it, and share creative household reuse ideas!"
                    ),
                    "type": "no_detection_context",
                    "waste_info": None,
                }

        # ── Pass 2: Plastic type queries ──────────────────────────────────────
        plastic_type_match = self._check_plastic_type(message_clean)
        if plastic_type_match:
            return plastic_type_match

        # ── Pass 3: Specific item queries (battery, straw, mask, etc.) ────────
        item_match = self._check_item_knowledge(message_clean)
        if item_match:
            return item_match

        # ── Pass 4: Decomposition / how long questions ────────────────────────
        if any(kw in message_lower for kw in [
            "how long", "decompose", "decomposition", "degrade", "biodegradable",
            "break down", "breakdown", "decay", "how many years",
        ]):
            # Check if they're asking about a specific waste type
            waste_key = self._find_waste_key(message_clean)
            if waste_key and waste_key in WASTE_KNOWLEDGE:
                info = WASTE_KNOWLEDGE[waste_key]
                impact = info["impact"]
                return {
                    "message": (
                        f"{info['icon']} **Decomposition — {info['category']}**\n\n"
                        f"⏳ **Time to decompose:** {impact['decomposition_years']}\n\n"
                        f"💡 {impact['fact']}\n\n"
                        "Want to know more? Ask:\n"
                        f"• *'How to dispose of {waste_key.replace('_', ' ')}?'*\n"
                        f"• *'Can {waste_key.replace('_', ' ')} be recycled?'*"
                    ),
                    "type": "decomposition",
                    "waste_info": {"waste_type": waste_key, "color": info["color"]},
                }
            # General decomposition table
            return {
                "message": FAQ_RESPONSES["decomposition"],
                "type": "faq",
                "waste_info": None,
            }

        # ── Pass 5: Direct topic keywords (must be before alias matching) ──
        if any(kw in message_lower for kw in ["compost", "vermicompost", "bokashi"]):
            return {"message": FAQ_RESPONSES["composting"], "type": "faq", "waste_info": None}

        if any(kw in message_lower for kw in ["zero waste", "zero-waste"]):
            return {"message": FAQ_RESPONSES["zero waste"], "type": "faq", "waste_info": None}

        if any(kw in message_lower for kw in ["microplastic", "micro plastic", "microbead"]):
            return {"message": FAQ_RESPONSES["microplastic"], "type": "faq", "waste_info": None}

        if any(kw in message_lower for kw in ["swachh", "swachhata", "clean india"]):
            return {"message": FAQ_RESPONSES["swachh bharat"], "type": "faq", "waste_info": None}

        if any(kw in message_lower for kw in [
            "reduce plastic", "less plastic", "avoid plastic", "plastic free",
            "plastic-free", "ban plastic",
        ]):
            return {"message": FAQ_RESPONSES["reduce plastic"], "type": "faq", "waste_info": None}

        # ── Pass 5.5: Knowledge Encyclopedia Match ────────────────────────────
        for topic_key, data in KNOWLEDGE_ENCYCLOPEDIA.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', message_lower) for kw in data["keywords"]):
                message = (
                    f"🌐 **AI Knowledge & Web Summary: {data['title']}**\n\n"
                    f"📌 **What It Means (In Simple Words):**\n"
                    f"{data['simple_explanation']}\n\n"
                    f"♻️ **Can It Be Recycled or Reused?**\n"
                    f"{data['recyclable_status']}\n\n"
                    f"🏡 **Household Action Steps:**\n"
                    f"{data['household_advice']}\n\n"
                    f"🌍 **Environmental Takeaway:**\n"
                    f"{data['eco_impact']}"
                )
                return {
                    "message": message,
                    "type": "web_knowledge_summary",
                    "waste_info": None,
                }

        # ── Pass 6: Waste type query with intent detection ────────────────────
        waste_key = self._find_waste_key(message_clean)
        if waste_key and waste_key in WASTE_KNOWLEDGE:
            info = WASTE_KNOWLEDGE[waste_key]
            # Detect intent
            if any(kw in message_lower for kw in [
                "dispose", "throw", "discard", "get rid", "bin", "where do i put",
                "where to put", "which bin", "how to throw", "how do i throw",
                "how should i", "what do i do with",
            ]):
                return self._format_disposal(waste_key, info)
            elif any(kw in message_lower for kw in [
                "reuse", "reused", "repurpose", "repurposing", "upcycle", "upcycling",
                "how to reuse", "how can i reuse", "can it be reused", "ways to reuse",
            ]):
                return self._format_reuse(waste_key, info)
            elif any(kw in message_lower for kw in [
                "recycle", "recyclable", "can i recycle",
                "is it recyclable", "recycled into",
            ]):
                return self._format_recycling(waste_key, info)
            elif any(kw in message_lower for kw in [
                "impact", "environment", "carbon", "eco", "pollution",
                "footprint", "climate", "emission",
            ]):
                return self._format_impact(waste_key, info)
            elif any(kw in message_lower for kw in [
                "tip", "tips", "reduce", "alternative", "avoid", "minimize",
                "less", "sustainable", "eco-friendly",
            ]):
                return self._format_tips(waste_key, info)
            elif any(kw in message_lower for kw in [
                "what is", "what are", "tell me about", "info", "information",
                "explain", "about", "describe",
            ]):
                return self._format_full_info(waste_key, info)
            else:
                # Default: give full info if they just mention the waste type
                return self._format_full_info(waste_key, info)

        # ── Pass 7: General topic questions ────────────────────────────────────
        if any(kw in message_lower for kw in ["segregat", "separate", "sort", "sorting"]):
            return {"message": FAQ_RESPONSES["what is waste segregation"], "type": "faq", "waste_info": None}

        if any(kw in message_lower for kw in ["why recycle", "importance of recycl", "benefit of recycl"]):
            return {"message": FAQ_RESPONSES["why recycle"], "type": "faq", "waste_info": None}

        # ── Pass 7.5: FAQ matching (fuzzy keyword match) ──────────────────────
        faq_match = self._match_faq(message_clean)
        if faq_match:
            return faq_match

        # ── Pass 8: Generic disposal question ─────────────────────────────────
        if any(kw in message_lower for kw in [
            "how to dispose", "how do i dispose", "how to throw",
            "where to throw", "where do i throw", "which bin",
        ]):
            return {
                "message": (
                    "I'd love to help with disposal! Could you tell me **what item** you want to dispose of?\n\n"
                    "For example:\n"
                    "• *'How to dispose of plastic bottles?'*\n"
                    "• *'Where to throw batteries?'*\n"
                    "• *'Which bin for food waste?'*\n"
                    "• *'How to dispose of old clothes?'*\n\n"
                    "Or use the **📷 Detect** tab to upload a photo and I'll identify it!"
                ),
                "type": "clarification",
                "waste_info": None,
            }

        # ── Fallback — smart, not generic ─────────────────────────────────────
        return self._smart_fallback(original)

    # ─── Matching Helpers ────────────────────────────────────────────────────────

    def _find_waste_key(self, text: str) -> Optional[str]:
        """Find which waste category a message is about using ALIAS_MAP."""
        best_match = None
        best_score = 0
        words = text.split()

        for waste_key, aliases in ALIAS_MAP.items():
            for alias in aliases:
                alias_words = alias.split()

                # For short aliases (<=3 chars), require whole-word match
                # to prevent false positives like 'tin' in 'composting'
                if len(alias) <= 3:
                    if not all(aw in words for aw in alias_words):
                        continue

                if alias in text:
                    # Longer alias matches = higher confidence
                    score = len(alias)
                    # Bonus for exact word match (not substring)
                    if all(aw in words for aw in alias_words):
                        score += 10
                    if score > best_score:
                        best_score = score
                        best_match = waste_key

        return best_match

    def _check_plastic_type(self, text: str) -> Optional[Dict]:
        """Check if the user is asking about a specific plastic type."""
        plastic_type_keywords = {
            "pet": ["pet plastic", "pet 1", "type 1", "#1 plastic", "number 1 plastic",
                    "polyethylene terephthalate"],
            "hdpe": ["hdpe", "type 2", "#2 plastic", "number 2 plastic",
                     "high density polyethylene", "high-density polyethylene"],
            "pvc": ["pvc", "type 3", "#3 plastic", "number 3 plastic",
                    "polyvinyl chloride", "vinyl"],
            "ldpe": ["ldpe", "type 4", "#4 plastic", "number 4 plastic",
                     "low density polyethylene", "low-density polyethylene"],
            "pp": ["polypropylene", "type 5", "#5 plastic", "number 5 plastic"],
            "ps": ["polystyrene", "type 6", "#6 plastic", "number 6 plastic",
                   "styrofoam", "thermocol"],
            "other": ["type 7", "#7 plastic", "number 7 plastic", "polycarbonate",
                      "bpa", "nylon plastic", "abs plastic"],
        }

        # Special: "pp" is too short, only match if preceded by word boundary
        for ptype, keywords in plastic_type_keywords.items():
            for kw in keywords:
                if kw in text:
                    info = PLASTIC_TYPES[ptype]
                    recyclable_text = "✅ **Yes, recyclable**" if info["recyclable"] else "❌ **Not easily recyclable**"
                    return {
                        "message": (
                            f"🛢️ **{info['code']}**\n\n"
                            f"**Common items:** {info['common_items']}\n\n"
                            f"**Recyclable?** {recyclable_text}\n"
                            f"**Recycled into:** {info['recycled_into']}\n\n"
                            f"**Decomposition time:** {info['decomposition']}\n\n"
                            f"**Health/Safety:** {info['health_risk']}\n\n"
                            f"💡 **Tip:** {info['tip']}"
                        ),
                        "type": "plastic_type",
                        "waste_info": {"waste_type": "plastic", "plastic_code": ptype},
                    }

        # General "types of plastic" question
        if any(kw in text for kw in [
            "types of plastic", "plastic types", "plastic codes", "plastic numbers",
            "recycling numbers", "recycling symbols", "resin codes",
            "what type of plastic", "kinds of plastic",
        ]):
            lines = []
            for ptype, info in PLASTIC_TYPES.items():
                symbol = "✅" if info["recyclable"] else "❌"
                lines.append(f"  {symbol} **{info['code']}** — {info['common_items'].split(',')[0]}")
            all_types = "\n".join(lines)
            return {
                "message": (
                    "🛢️ **The 7 Types of Plastic (Resin Identification Codes):**\n\n"
                    f"{all_types}\n\n"
                    "✅ = Commonly recyclable | ❌ = Difficult to recycle\n\n"
                    "Ask about any specific type for detailed info, e.g. *'What is PET plastic?'*"
                ),
                "type": "plastic_types",
                "waste_info": {"waste_type": "plastic"},
            }

        return None

    def _check_item_knowledge(self, text: str) -> Optional[Dict]:
        """Check if asking about a specific item with dedicated knowledge."""
        item_triggers = {
            "bottle": ["bottle", "bottles"],
            "battery": ["battery", "batteries", "cell battery", "button cell"],
            "straw": ["straw", "straws", "drinking straw"],
            "mask": ["mask", "masks", "face mask", "surgical mask", "n95"],
            "diaper": ["diaper", "diapers", "nappy", "nappies"],
        }

        for item_key, triggers in item_triggers.items():
            for trigger in triggers:
                if trigger in text:
                    info = ITEM_KNOWLEDGE[item_key]
                    return {
                        "message": info["answer"],
                        "type": "item_info",
                        "waste_info": {"item": item_key},
                    }
        return None

    def _match_faq(self, text: str) -> Optional[Dict]:
        """Fuzzy match against FAQ responses using keyword overlap."""
        best_match = None
        best_score = 0.0

        for faq_key, faq_response in FAQ_RESPONSES.items():
            keywords = faq_key.split()
            if len(keywords) == 0:
                continue

            # Count how many FAQ keywords appear in the user message
            matches = sum(1 for kw in keywords if kw in text)
            score = matches / len(keywords)

            # Also check if the full FAQ key is a substring
            if faq_key in text:
                score = 1.0

            if score > best_score and score >= 0.7:
                best_score = score
                best_match = faq_response

        if best_match:
            return {
                "message": best_match,
                "type": "faq",
                "waste_info": None,
            }
        return None

    def _analyze_and_summarize_knowledge(self, query: str) -> Optional[Dict]:
        """
        Analyze and synthesize knowledge from the curated encyclopedia or online resources,
        providing a structured, easy-to-understand summary for everyone.
        """
        query_lower = query.lower()

        # 1. Curated Encyclopedia match
        for topic_key, data in KNOWLEDGE_ENCYCLOPEDIA.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', query_lower) for kw in data["keywords"]):
                message = (
                    f"🌐 **AI Knowledge & Web Summary: {data['title']}**\n\n"
                    f"📌 **What It Means (In Simple Words):**\n"
                    f"{data['simple_explanation']}\n\n"
                    f"♻️ **Can It Be Recycled or Reused?**\n"
                    f"{data['recyclable_status']}\n\n"
                    f"🏡 **Household Action Steps:**\n"
                    f"{data['household_advice']}\n\n"
                    f"🌍 **Environmental Takeaway:**\n"
                    f"{data['eco_impact']}"
                )
                return {
                    "message": message,
                    "type": "web_knowledge_summary",
                    "waste_info": None,
                }

        # 2. Try online web search via Wikipedia REST API
        web_res = fetch_web_knowledge_summary(query)
        if web_res and web_res.get("extract"):
            title = web_res["title"]
            extract = web_res["extract"]

            # Clean and truncate extract to ~380 chars for clean readability
            if len(extract) > 380:
                extract = extract[:380].rsplit(".", 1)[0] + "."

            message = (
                f"🌐 **AI Knowledge & Web Summary: {title}**\n\n"
                f"📌 **What It Means (In Simple Words):**\n"
                f"{extract}\n\n"
                f"♻️ **Recycling & Waste Management Insight:**\n"
                f"• Clean, dry materials should be segregated into the **BLUE bin (Dry Recyclables)**.\n"
                f"• Organic and biodegradable food items belong in the **GREEN bin (Wet Waste)**.\n"
                f"• When possible, aim to reuse or upcycle items at home before discarding them!\n\n"
                f"💡 **Eco Tip for Everyone:**\n"
                f"Reducing single-use consumption and segregating at source prevents up to 80% of waste from reaching landfills."
            )
            return {
                "message": message,
                "type": "web_knowledge_summary",
                "waste_info": None,
            }

        return None

    def _smart_fallback(self, original: str) -> Dict:
        """Smart fallback that uses ExecuTorch Llama engine before falling back."""
        # 1. Try ExecuTorch Llama 3.2 engine if available
        if self.llama_engine:
            try:
                llama_res = self.llama_engine.generate(original)
                if llama_res and llama_res.get("text"):
                    return {
                        "message": (
                            f"{llama_res['text']}\n\n"
                            f"_⚡ ExecuTorch On-Device Engine ({llama_res['engine']}) | {llama_res['inference_time_ms']} ms_"
                        ),
                        "type": "llama_executorch",
                        "waste_info": {"engine": llama_res["engine"]},
                    }
            except Exception as e:
                logger.warning(f"Llama engine generation error: {e}")

        # 2. Try web and knowledge synthesis
        knowledge_res = self._analyze_and_summarize_knowledge(original)
        if knowledge_res:
            return knowledge_res

        msg = original.lower()

        # If they're asking a question at all, guide them
        if "?" in original or any(msg.startswith(w) for w in [
            "how", "what", "where", "which", "can", "is", "why", "when", "do", "does", "will",
        ]):
            return {
                "message": (
                    f"That's a great question! I specialise in **waste management, recycling, and disposal**.\n\n"
                    "Here are topics I can help with:\n\n"
                    "🗑️ **Disposal:** *'How to dispose of [item]?'*\n"
                    "♻️ **Recycling:** *'Can [item] be recycled?'*\n"
                    "🛢️ **Plastic types:** *'What is PET/HDPE/PVC plastic?'*\n"
                    "⏳ **Decomposition:** *'How long does [item] take to decompose?'*\n"
                    "🌍 **Environment:** *'Environmental impact of plastic'*\n"
                    "🌱 **Composting:** *'How to start composting?'*\n"
                    "🏠 **Zero waste:** *'What is zero waste?'*\n"
                    "📷 **Detection:** Upload a photo in the **Detect** tab!\n\n"
                    "Try asking about a specific material or upload an item to scan!"
                ),
                "type": "fallback",
                "waste_info": None,
            }

        return {
            "message": (
                "I'm your **AI Waste Disposal Assistant** 🤖\n\n"
                "I can answer questions about:\n"
                "• 🗑️ How to dispose of any waste material\n"
                "• ♻️ What can be recycled and how\n"
                "• 🛢️ Types of plastic (#1-#7) and their properties\n"
                "• ⏳ How long different items take to decompose\n"
                "• 🌍 Environmental impact of waste\n"
                "• 🌱 Composting and zero-waste tips\n\n"
                "**Try asking:**\n"
                "• *'How to dispose of plastic bottles?'*\n"
                "• *'What type of plastic is a water bottle?'*\n"
                "• *'How long does styrofoam take to decompose?'*\n"
                "• *'Where to throw batteries?'*\n"
                "• *'What is waste segregation?'*"
            ),
            "type": "fallback",
            "waste_info": None,
        }

    def _help_response(self) -> Dict:
        """Respond to help/menu requests."""
        return {
            "message": (
                "📚 **AI Waste Advisor — Help Menu**\n\n"
                "Here's everything I can help you with:\n\n"
                "**🗑️ Disposal Guidance**\n"
                "  Ask *'How to dispose of [item]?'* for any item — plastic, paper, food, batteries, electronics, clothes, etc.\n\n"
                "**♻️ Recycling Information**\n"
                "  Ask *'Can [item] be recycled?'* to learn recycling options.\n\n"
                "**🛢️ Plastic Types (Resin Codes #1-#7)**\n"
                "  Ask *'Types of plastic'* or *'What is PET/HDPE/PVC?'* for detailed info on each type.\n\n"
                "**⏳ Decomposition Times**\n"
                "  Ask *'How long does [item] decompose?'* or just *'decomposition'*.\n\n"
                "**🌍 Environmental Impact**\n"
                "  Ask *'Impact of plastic'* or *'Environmental impact of e-waste'*.\n\n"
                "**🌱 Eco-Living Tips**\n"
                "  Try *'composting'*, *'zero waste'*, *'reduce plastic'*, *'Swachh Bharat'*.\n\n"
                "**📷 Waste Detection**\n"
                "  Upload a photo in the **Detect** tab — I'll identify the waste and give disposal advice!\n\n"
                "**💡 Quick topics:** plastic, paper, glass, metal, food waste, battery, e-waste, mask, diaper, straw"
            ),
            "type": "help",
            "waste_info": None,
        }

    # ─── Response Formatters ─────────────────────────────────────────────────────

    def _format_disposal(self, waste_key: str, info: Dict) -> Dict:
        icon = info["icon"]
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(info["disposal"]))
        tip = info["tips"][0] if info["tips"] else ""
        return {
            "message": (
                f"{icon} **Disposal Guide — {info['category']}**\n\n"
                f"{steps}\n\n"
                f"💚 **Tip:** {tip}"
            ),
            "type": "disposal",
            "waste_info": {"waste_type": waste_key, "color": info["color"]},
        }

    def _format_recycling(self, waste_key: str, info: Dict) -> Dict:
        icon = info["icon"]
        items = "\n".join(f"  • {r}" for r in info["recycling"])
        return {
            "message": (
                f"{icon} **Recycling Options — {info['category']}**\n\n"
                f"{items}\n\n"
                f"🌍 **Impact:** {info['impact']['fact']}"
            ),
            "type": "recycling",
            "waste_info": {"waste_type": waste_key, "color": info["color"]},
        }

    def _format_impact(self, waste_key: str, info: Dict) -> Dict:
        impact = info["impact"]
        return {
            "message": (
                f"🌍 **Environmental Impact — {info['category']}**\n\n"
                f"  • ⏳ Decomposition: {impact['decomposition_years']}\n"
                f"  • 🌱 CO₂ saved by recycling: {impact['co2_saved_kg']} kg\n"
                f"  • 💧 Water saved: {impact['water_saved_liters']} litres\n"
                f"  • ⚡ Energy saved: {impact['energy_saved_kwh']} kWh\n\n"
                f"  💡 **Did you know?** {impact['fact']}"
            ),
            "type": "impact",
            "waste_info": {"waste_type": waste_key, "color": info["color"]},
        }

    def _format_tips(self, waste_key: str, info: Dict) -> Dict:
        tips = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(info["tips"]))
        return {
            "message": f"💚 **Sustainability Tips — {info['category']}**\n\n{tips}",
            "type": "tips",
            "waste_info": {"waste_type": waste_key, "color": info["color"]},
        }

    def _format_reuse(self, waste_key: str, info: Dict) -> Dict:
        icon = info["icon"]
        reuse_items = info.get("reuse", [])
        if reuse_items:
            items_str = "\n".join(f"  • {r}" for r in reuse_items)
        else:
            items_str = "  • Consider repurposing or donating usable items before discarding."

        return {
            "message": (
                f"{icon} **Creative Reuse & Upcycling — {info['category']}**\n\n"
                f"Here are practical ways to give your {waste_key.replace('_', ' ')} a second life:\n\n"
                f"{items_str}\n\n"
                f"🌍 **Eco Benefit:** Reusing items directly prevents waste from reaching landfills and reduces the need for new manufacturing!"
            ),
            "type": "reuse",
            "waste_info": {"waste_type": waste_key, "color": info["color"]},
        }

    def _format_complete_analysis_for_detection(
        self,
        waste_type: str,
        confidence: float = 0.0,
        bbox: Optional[List[float]] = None,
        detected_object: Optional[str] = None,
    ) -> Dict:
        analysis = self.guess_object_and_analysis(waste_type, confidence, bbox, detected_object=detected_object)
        waste_type_key = waste_type.lower().replace(" ", "_").replace("-", "_")
        info = WASTE_KNOWLEDGE.get(waste_type_key, WASTE_KNOWLEDGE["plastic"])

        recycling_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(analysis["recycling_steps"]))
        reuse_text = "\n".join(f"  • {h}" for h in analysis["household_reuse"])

        message = (
            f"🎯 **AI Object Guess & Comprehensive Analysis** (Confidence: {confidence:.0%})\n\n"
            f"• **Guessed Object:** {analysis['guessed_object']}\n"
            f"• **Likely Material / Resin:** {analysis['plastic_type']}\n\n"
            f"**♻️ How the Object Can Be Recycled:**\n{recycling_text}\n\n"
            f"**🏡 How It Can Be Reused for Household Purpose (DIY Upcycling):**\n{reuse_text}\n\n"
            f"**⏳ Decomposition Time:** {analysis['decomposition_time']} | 💡 {info['impact']['fact']}"
        )

        return {
            "message": message,
            "type": "complete_detection_analysis",
            "waste_info": {
                "waste_type": waste_type_key,
                "category": info["category"],
                "color": info["color"],
                "object_guess": analysis["guessed_object"],
                "plastic_type": analysis["plastic_type"],
                "recycling_steps": analysis["recycling_steps"],
                "household_reuse": analysis["household_reuse"],
            },
        }

    def _format_household_reuse_for_detection(
        self,
        waste_type: str,
        confidence: float = 0.0,
        bbox: Optional[List[float]] = None,
        detected_object: Optional[str] = None,
    ) -> Dict:
        analysis = self.guess_object_and_analysis(waste_type, confidence, bbox, detected_object=detected_object)
        waste_type_key = waste_type.lower().replace(" ", "_").replace("-", "_")
        info = WASTE_KNOWLEDGE.get(waste_type_key, WASTE_KNOWLEDGE["plastic"])

        reuse_text = "\n".join(f"  • {h}" for h in analysis["household_reuse"])

        message = (
            f"✅ **Yes, absolutely! It can definitely be reused and repurposed.**\n\n"
            f"🏡 **Household Reuse & DIY Upcycling Guide** (Detected: **{analysis['guessed_object']}**, Confidence: {confidence:.0%})\n\n"
            f"Here are practical, creative ways to reuse this object at home:\n\n"
            f"{reuse_text}\n\n"
            f"💡 **Tip:** Always wash and dry the item thoroughly before repurposing for household storage or crafts!"
        )

        return {
            "message": message,
            "type": "household_reuse",
            "waste_info": {
                "waste_type": waste_type_key,
                "category": info["category"],
                "color": info["color"],
                "object_guess": analysis["guessed_object"],
                "household_reuse": analysis["household_reuse"],
            },
        }

    def _format_reuse_for_detection(
        self,
        waste_type: str,
        confidence: float = 0.0,
        bbox: Optional[List[float]] = None,
        detected_object: Optional[str] = None,
    ) -> Dict:
        return self._format_household_reuse_for_detection(waste_type, confidence, bbox, detected_object=detected_object)

    def _format_recycling_for_detection(
        self,
        waste_type: str,
        confidence: float = 0.0,
        bbox: Optional[List[float]] = None,
        detected_object: Optional[str] = None,
    ) -> Dict:
        analysis = self.guess_object_and_analysis(waste_type, confidence, bbox, detected_object=detected_object)
        waste_type_key = waste_type.lower().replace(" ", "_").replace("-", "_")
        info = WASTE_KNOWLEDGE.get(waste_type_key, WASTE_KNOWLEDGE["plastic"])

        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(analysis["recycling_steps"]))

        message = (
            f"♻️ **Yes, this object is recyclable!**\n\n"
            f"**How the Object Can Be Recycled** (Detected: **{analysis['guessed_object']}**, Confidence: {confidence:.0%})\n\n"
            f"• **Material & Resin Code:** {analysis['plastic_type']}\n\n"
            f"**Disposal Steps:**\n{steps_text}\n\n"
            f"🌍 **Environmental Impact:** Recycling prevents {analysis['decomposition_time']} of landfill persistence!"
        )

        return {
            "message": message,
            "type": "detection_recycling",
            "waste_info": {
                "waste_type": waste_type_key,
                "category": info["category"],
                "color": info["color"],
                "recycling_steps": analysis["recycling_steps"],
            },
        }

    def _format_object_guess_for_detection(
        self,
        waste_type: str,
        confidence: float = 0.0,
        bbox: Optional[List[float]] = None,
        detected_object: Optional[str] = None,
    ) -> Dict:
        analysis = self.guess_object_and_analysis(waste_type, confidence, bbox, detected_object=detected_object)
        waste_type_key = waste_type.lower().replace(" ", "_").replace("-", "_")
        info = WASTE_KNOWLEDGE.get(waste_type_key, WASTE_KNOWLEDGE["plastic"])

        message = (
            f"🎯 **AI Object & Material Identification** (Confidence: {confidence:.0%})\n\n"
            f"• **Most Likely Object:** {analysis['guessed_object']}\n"
            f"• **Material / Plastic Code:** {analysis['plastic_type']} ({analysis['resin_code']})\n"
            f"• **Decomposition Time:** {analysis['decomposition_time']}\n\n"
            f"Ask me: *'How can it be reused for household purpose?'* or *'How can this object be recycled?'* for more!"
        )

        return {
            "message": message,
            "type": "object_guess",
            "waste_info": {
                "waste_type": waste_type_key,
                "category": info["category"],
                "color": info["color"],
                "object_guess": analysis["guessed_object"],
            },
        }

    def _format_impact_for_detection(self, waste_type: str, confidence: float = 0.0) -> Dict:
        waste_type_key = waste_type.lower().replace(" ", "_").replace("-", "_")
        info = WASTE_KNOWLEDGE.get(waste_type_key, WASTE_KNOWLEDGE["plastic"])
        impact = info["impact"]

        return {
            "message": (
                f"⏳ **Decomposition & Impact for Detected Item** (Detected: **{waste_type.replace('_', ' ')}**, Confidence: {confidence:.0%})\n\n"
                f"• **Category:** {info['category']}\n"
                f"• **Decomposition time:** {impact['decomposition_years']}\n"
                f"• **CO₂ saved by recycling:** {impact['co2_saved_kg']} kg\n"
                f"• **Water saved:** {impact['water_saved_liters']} litres\n\n"
                f"💡 **Fact:** {impact['fact']}"
            ),
            "type": "detection_impact",
            "waste_info": {"waste_type": waste_type_key, "color": info["color"]},
        }

    def _format_full_info(self, waste_key: str, info: Dict) -> Dict:
        icon = info["icon"]
        disposal = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(info["disposal"]))
        recycling = "\n".join(f"  • {r}" for r in info["recycling"][:3])
        impact = info["impact"]
        tip = info["tips"][0] if info["tips"] else ""

        return {
            "message": (
                f"{icon} **{info['category']}**\n\n"
                f"**🗑️ How to Dispose:**\n{disposal}\n\n"
                f"**♻️ Recycling Options:**\n{recycling}\n\n"
                f"**🌍 Impact:**\n"
                f"  • Decomposition: {impact['decomposition_years']}\n"
                f"  • {impact['fact']}\n\n"
                f"**💚 Tip:** {tip}"
            ),
            "type": "full_info",
            "waste_info": {"waste_type": waste_key, "color": info["color"]},
        }
