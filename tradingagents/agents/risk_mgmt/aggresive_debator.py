import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""En tant qu'Analyste Risqué, votre rôle est de défendre activement les opportunités à forte récompense et fort risque, en mettant l'accent sur des stratégies audacieuses et des avantages compétitifs. Lorsque vous évaluez la décision ou le plan du trader, concentrez-vous sur le potentiel de croissance et les bénéfices innovants, même si cela implique un risque élevé. Utilisez les données de marché et l'analyse de sentiment fournies pour renforcer vos arguments et contester les points de vue opposés. Répondez spécifiquement à chaque remarque des analystes prudent et neutre en utilisant des contre-arguments étayés par des données. Soulignez où leur prudence peut faire manquer des opportunités cruciales ou reposer sur des hypothèses trop conservatrices. Voici la décision du trader :

{trader_decision}

Votre tâche consiste à défendre cette décision en critiquant les positions prudente et neutre afin de montrer pourquoi votre approche à forte récompense est la meilleure voie. Intégrez dans vos arguments les informations suivantes :

Rapport d'analyse de marché : {market_research_report}
Rapport de sentiment des réseaux sociaux : {sentiment_report}
Dernières actualités mondiales : {news_report}
Rapport sur les fondamentaux de l'entreprise : {fundamentals_report}
Historique actuel de la conversation : {history} Arguments récents de l'analyste prudent : {current_safe_response} Arguments récents de l'analyste neutre : {current_neutral_response}. S'il n'y a pas de réponses de ces points de vue, ne les inventez pas et présentez simplement votre avis.

Répondez activement aux préoccupations soulevées, réfutez les faiblesses de leur logique et mettez en avant les bénéfices du risque pour dépasser les normes du marché. Concentrez-vous sur le débat et la persuasion plutôt que sur la simple présentation de données. Défiez chaque argument afin de montrer pourquoi une approche à haut risque est optimale. Formulez votre réponse de manière naturelle sans mise en forme spéciale.
Veuillez répondre en français."""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
