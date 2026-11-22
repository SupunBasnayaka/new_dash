import os

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "paddy_data.db")

# The model needs to know how this data is laid out, or it will happily double-count
# (Total = Major + Minor + Rainfed) or add the national row to the districts.
AGENT_PREFIX = """You are a data analyst for Sri Lankan paddy statistics (Dept. of Census & Statistics).
You query a SQLite database through SQL tools. Rules for correct answers:

TABLE paddy_stats - one row per (district, season, year, irrigation_type).
- area_type: 'district' (25 districts), 'scheme' (Udawalawe, Mahaweli H, Highland Paddy) or
  'national' (the 'Sri Lanka' row). For district or province questions filter area_type = 'district'.
  Never add national or scheme rows to district rows.
- irrigation_type: 'Major', 'Minor', 'Rainfed' and 'Total'. Total already equals the other three
  combined, so filter irrigation_type = 'Total' unless the question is about irrigation type.
  production_mt exists ONLY on 'Total' rows (it is NULL for Major/Minor/Rainfed).
- season: 'Maha' (Oct-Mar) or 'Yala' (Apr-Sep). year is the calendar year the season ends in
  (Maha 2023/2024 has year 2024). season_label shows the original label.
- Units: sown_extent_ha and harvested_extent_ha in hectares, average_yield_kg_ha in kg/ha, production_mt in metric tonnes.
- NULL means the source had no value; never treat NULL as zero. Averages of yield across
  districts should be weighted by harvested_extent_ha when asked for a national/provincial yield.
- Maha 2026 (2025/2026) has no production data yet. Irrigation-type splits only exist from 1984.
- province is set for districts only (9 provinces).

TABLE national_paddy_stats - official national season totals 1952-2026 in original units
(000 acres, 000 ha, bushels/acre, kg/ha, 000 bushels, 000 Mt).

Always state the years, season and units used. If data is missing, say so instead of guessing."""


def get_paddy_sql_agent(openrouter_api_key: str):
    if not openrouter_api_key:
        return None

    # Read-only connection: the assistant can never modify the data.
    db = SQLDatabase.from_uri(
        f"sqlite:///file:{DB_PATH}?mode=ro&uri=true",
        include_tables=["paddy_stats", "national_paddy_stats"],
        sample_rows_in_table_info=2,
    )

    llm = ChatOpenAI(
        model="deepseek/deepseek-chat",
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        prefix=AGENT_PREFIX,
        verbose=False,
        agent_type="openai-tools",
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def ask_agent(agent, question: str) -> str:
    """Run one question and return the final answer text."""
    result = agent.invoke({"input": question})
    return result["output"] if isinstance(result, dict) else str(result)
