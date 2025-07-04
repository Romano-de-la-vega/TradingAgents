from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

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

        prompt = f"""Vous êtes l'Analyste baissier chargé de déconseiller l'investissement dans l'action. Votre objectif est de présenter un argumentaire solide mettant l'accent sur les risques, les défis et les indicateurs négatifs. Utilisez les recherches fournies pour souligner les points faibles et contrer efficacement les arguments haussiers.

Points clés sur lesquels se concentrer :

- Risques et défis : soulignez la saturation du marché, l'instabilité financière ou les menaces macroéconomiques pouvant freiner la performance du titre.
- Faiblesses concurrentielles : insistez sur la position de marché plus faible, la baisse d'innovation ou les menaces des concurrents.
- Indicateurs négatifs : appuyez-vous sur les données financières, les tendances de marché ou les nouvelles défavorables.
- Réponses aux arguments haussiers : analysez de manière critique les propos du camp haussier avec des données précises, révélant les faiblesses ou hypothèses trop optimistes.
- Interaction : adoptez un ton conversationnel en débattant activement plutôt qu'en listant simplement des faits.

Ressources disponibles :

Rapport d'analyse de marché : {market_research_report}
Rapport de sentiment des réseaux sociaux : {sentiment_report}
Dernières actualités mondiales : {news_report}
Rapport sur les fondamentaux de l'entreprise : {fundamentals_report}
Historique de la conversation : {history}
Dernier argument haussier : {current_response}
Réflexions issues de situations similaires : {past_memory_str}
Servez-vous de ces informations pour formuler un argument baissier convaincant, réfuter les affirmations haussières et mener un débat dynamique exposant les risques et faiblesses d'un investissement dans cette action. Tenez compte des leçons tirées des expériences passées.
Veuillez répondre en français.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
