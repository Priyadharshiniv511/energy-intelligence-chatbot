"""What the bot will and won't talk about: the single source of truth (DESIGN.md 2.5).

WHY one module: the classifier prompt, the answerer prompt (G3) and the adversarial
tests all need the same scope. If each kept its own copy, they would drift apart, and
a topic could be "allowed" by the prompt but "refused" by the tests (or vice versa).
Change scope here and every consumer follows.

This module holds data only. How it is phrased for a model lives in prompts.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicArea:
    """One scope entry.

    `key` is a stable id for tests and logs. `description` is what the model reads.
    `example` is a typical question in this area; the adversarial suite uses it as a test
    case, so every IN and OUT item is exercised at least once (DESIGN.md 2.5).
    """

    key: str
    description: str
    example: str


IN_SCOPE: tuple[TopicArea, ...] = (
    TopicArea("renewables", "Renewable energy and solar: PV, wind, hydro, geothermal, storage",
              "How much electricity does a 5 kW rooftop solar system produce per year?"),
    TopicArea("efficiency", "Energy efficiency and optimisation of homes, buildings and processes",
              "What are the most effective ways to cut my home's heating energy use?"),
    TopicArea("generation_distribution",
              "Electricity generation, transmission and distribution (power plants, lines, substations)",
              "Why is electricity transmitted at high voltage?"),
    TopicArea("grids_smart_meters", "Power grids, grid stability, demand response and smart meters",
              "What data does a smart meter record?"),
    TopicArea("industrial_commercial", "Industrial and commercial energy usage and management",
              "How can a factory reduce its peak electricity demand charges?"),
    TopicArea("billing", "Electricity and energy billing: reading bills, units, charges",
              "Why does my electricity bill have both a fixed charge and a per-kWh charge?"),
    TopicArea("ev_charging", "Electric vehicles and EV charging (energy use, charging levels, costs)",
              "How long does it take to charge an EV on a 7 kW home charger?"),
    TopicArea("prices_tariffs",
              "Energy prices and tariffs in general terms (how they work, not investment advice)",
              "How do time-of-use electricity tariffs work?"),
    TopicArea("emissions", "Greenhouse-gas emissions from energy generation and use",
              "How much CO2 does a coal power plant emit per kWh compared with gas?"),
    TopicArea("mining_datacentres", "Energy use of crypto-mining and data centres",
              "How much electricity do data centres use worldwide?"),
    TopicArea("appliance_efficiency", "Energy efficiency of household appliances",
              "How much electricity does a refrigerator use per year?"),
)

OUT_OF_SCOPE: tuple[TopicArea, ...] = (
    TopicArea("investment_advice",
              "Investment, trading or stock advice, including energy-company stocks and funds",
              "Which solar stock should I buy this year?"),
    TopicArea("food_energy", "Food calories, nutrition or 'food energy'",
              "How much energy is in a banana?"),
    TopicArea("non_power_physics",
              "Physics or homework problems not about power systems (e.g. kinetic energy of a ball)",
              "What is the kinetic energy of a 2 kg ball moving at 3 m/s?"),
    TopicArea("shopping_advice", "Brand or product shopping recommendations",
              "Which brand of solar panel should I buy, LG or Panasonic?"),
    TopicArea("coding_help", "Generic programming or coding help",
              "Write a Python function that sorts my energy readings."),
    TopicArea("everything_else", "Anything else outside the energy domain",
              "Who won the football World Cup in 2022?"),
)
