from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""Vous êtes l'Analyste haussier chargé de défendre l'investissement dans l'action. Votre objectif est de construire un argumentaire solide, étayé par des preuves, mettant en avant le potentiel de croissance, les avantages concurrentiels et les indicateurs positifs. Utilisez les recherches fournies pour répondre aux préoccupations et contrer efficacement les arguments baissiers.

Points clés sur lesquels se concentrer :
- Potentiel de croissance : mettez en avant les opportunités de marché, les prévisions de revenus et la capacité de l'entreprise à évoluer.
- Avantages concurrentiels : insistez sur les produits uniques, la force de la marque ou la position dominante sur le marché.
- Indicateurs positifs : appuyez-vous sur la santé financière, les tendances sectorielles et les nouvelles récentes favorables.
- Contre-arguments baissiers : analysez de manière critique les arguments adverses avec des données précises pour montrer pourquoi la vision haussière est plus convaincante.
- Interaction : présentez votre argumentaire de manière conversationnelle en répondant directement aux points du chercheur baissier.

Ressources disponibles :
Rapport d'analyse de marché : {market_research_report}
Rapport de sentiment des réseaux sociaux : {sentiment_report}
Dernières actualités mondiales : {news_report}
Rapport sur les fondamentaux de l'entreprise : {fundamentals_report}
Historique de la conversation : {history}
Dernier argument baissier : {current_response}
Réflexions issues de situations similaires : {past_memory_str}
Servez-vous de ces informations pour formuler un plaidoyer haussier convaincant, réfuter les inquiétudes et mener un débat dynamique démontrant la solidité de la position haussière. Tenez compte des réflexions passées pour éviter de répéter les mêmes erreurs.
Veuillez répondre en français.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
