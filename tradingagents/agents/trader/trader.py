import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Sur la base d'une analyse complète réalisée par une équipe d'analystes, voici un plan d'investissement adapté à {company_name}. Ce plan intègre les tendances techniques actuelles du marché, les indicateurs macroéconomiques et le sentiment des réseaux sociaux. Utilisez-le comme base pour évaluer votre prochaine décision de trading.\n\nPlan d'investissement proposé : {investment_plan}\n\nServez-vous de ces informations pour prendre une décision éclairée et stratégique.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""Vous êtes un agent de trading analysant les données de marché pour prendre des décisions d'investissement. Sur la base de votre analyse, fournissez une recommandation précise d'achat, de vente ou de conservation. Terminez par une décision ferme et concluez toujours votre réponse par 'PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL**' pour confirmer votre recommandation. N'oubliez pas d'utiliser les leçons des décisions passées pour apprendre de vos erreurs. Voici quelques réflexions issues de situations similaires : {past_memory_str}.
Veuillez répondre en français.""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
