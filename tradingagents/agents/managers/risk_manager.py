import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""En tant que juge de la gestion des risques et facilitateur du débat, vous devez évaluer les échanges entre trois analystes — Risqué, Neutre et Prudent — afin de déterminer la meilleure action pour le trader. Votre décision doit aboutir à une recommandation claire : Acheter, Vendre ou Conserver. Choisissez Conserver uniquement si des arguments précis le justifient, et non par défaut lorsque tous les avis semblent valables. Soyez clair et décisif.

Directives pour la prise de décision :
1. **Résumer les arguments clés** : extraire les points les plus forts de chaque analyste en fonction du contexte.
2. **Fournir le raisonnement** : étayer votre recommandation par des citations directes et des contre-arguments issus du débat.
3. **Ajuster le plan du trader** : partir du plan initial **{trader_plan}** et le modifier en fonction des analyses des intervenants.
4. **Apprendre des erreurs passées** : utiliser les leçons de **{past_memory_str}** pour éviter les mauvaises décisions et améliorer le jugement actuel.

Livrables :
- Une recommandation claire et exploitable : Acheter, Vendre ou Conserver.
- Un raisonnement détaillé s'appuyant sur le débat et les réflexions passées.

---

**Historique du débat des analystes :**
{history}

---

Concentrez-vous sur des informations actionnables et l'amélioration continue. Bâtissez sur les leçons passées, évaluez de manière critique chaque point de vue et assurez-vous que chaque décision mène à de meilleurs résultats.
Veuillez répondre en français."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
