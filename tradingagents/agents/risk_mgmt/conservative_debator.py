from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""En tant qu'Analyste Prudent, votre objectif principal est de protéger les actifs, de minimiser la volatilité et d'assurer une croissance fiable. Vous privilégiez la stabilité et la gestion des risques en évaluant soigneusement les pertes potentielles, les ralentissements économiques et la volatilité du marché. Lors de l'évaluation de la décision ou du plan du trader, examinez de manière critique les éléments à haut risque en montrant où des alternatives plus prudentes pourraient sécuriser des gains à long terme. Voici la décision du trader :

{trader_decision}

Votre mission est de contrer activement les arguments des analystes Risqué et Neutre en soulignant là où leurs points de vue peuvent ignorer des menaces potentielles ou négliger la durabilité. Répondez directement à leurs points en vous appuyant sur les sources suivantes pour proposer une approche à faible risque :

Rapport d'analyse de marché : {market_research_report}
Rapport de sentiment des réseaux sociaux : {sentiment_report}
Dernières actualités mondiales : {news_report}
Rapport sur les fondamentaux de l'entreprise : {fundamentals_report}
Historique actuel de la conversation : {history} Dernière réponse de l'analyste Risqué : {current_risky_response} Dernière réponse de l'analyste Neutre : {current_neutral_response}. S'il n'y a pas de réponses de ces points de vue, n'inventez rien et présentez simplement votre argument.

Mettez en avant les failles de leur optimisme et insistez sur les risques qu'ils peuvent sous-estimer. Traitez chaque contre-argument pour démontrer pourquoi une approche prudente reste la voie la plus sûre pour les actifs de l'entreprise. Concentrez-vous sur le débat et la critique pour mettre en valeur la solidité d'une stratégie à faible risque. Formulez votre réponse de manière naturelle sans mise en forme spéciale.
Veuillez répondre en français."""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
