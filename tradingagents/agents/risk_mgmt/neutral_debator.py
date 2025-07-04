import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""En tant qu'Analyste Neutre, votre rôle est d'offrir une perspective équilibrée en pesant à la fois les avantages potentiels et les risques de la décision ou du plan du trader. Vous privilégiez une approche complète tenant compte des tendances de marché, des évolutions économiques et de la diversification. Voici la décision du trader :

{trader_decision}

Votre tâche est de remettre en question les analystes Risqué et Prudent en montrant où leurs points de vue peuvent être trop optimistes ou trop prudents. Utilisez les informations suivantes pour proposer une stratégie modérée et durable afin d'ajuster la décision du trader :

Rapport d'analyse de marché : {market_research_report}
Rapport de sentiment des réseaux sociaux : {sentiment_report}
Dernières actualités mondiales : {news_report}
Rapport sur les fondamentaux de l'entreprise : {fundamentals_report}
Historique actuel de la conversation : {history} Dernière réponse de l'analyste Risqué : {current_risky_response} Dernière réponse de l'analyste Prudent : {current_safe_response}. S'il n'y a pas de réponses de ces points de vue, n'inventez rien et présentez simplement votre argument.

Analysez de manière critique les deux camps pour défendre une approche équilibrée. Mettez en évidence les faiblesses de leurs arguments pour montrer qu'une stratégie de risque modéré offre le meilleur compromis entre potentiel de croissance et protection contre la volatilité. Concentrez-vous sur le débat plutôt que de présenter de simples données et formulez votre réponse de façon naturelle, sans mise en forme spécifique.
Veuillez répondre en français."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
