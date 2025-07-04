import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""En tant que gestionnaire de portefeuille et modérateur du débat, votre rôle est d'évaluer de façon critique cette session et de prendre une décision définitive : vous ranger du côté de l'analyste baissier, de l'analyste haussier, ou choisir Conserver uniquement si les arguments le justifient clairement.

Résumez de manière concise les points clés de chaque partie en vous appuyant sur les preuves les plus convaincantes. Votre recommandation — Acheter, Vendre ou Conserver — doit être claire et immédiatement actionnable. Ne choisissez pas Conserver par défaut si les deux avis semblent valables ; privilégiez la position soutenue par les arguments les plus forts.

Élaborez ensuite un plan d'investissement détaillé pour le trader comprenant :
- Votre recommandation accompagnée des arguments principaux.
- Le raisonnement expliquant cette conclusion.
- Les actions stratégiques concrètes pour mettre en œuvre cette recommandation.
Tenez compte de vos erreurs passées dans des situations similaires pour améliorer votre prise de décision. Présentez votre analyse de manière naturelle, sans mise en forme particulière.

Voici vos réflexions précédentes sur les erreurs :
"{past_memory_str}"

Voici le débat :
Historique du débat :
{history}
Veuillez répondre en français."""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
