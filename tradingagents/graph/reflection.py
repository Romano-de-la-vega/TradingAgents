# TradingAgents/graph/reflection.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
Vous êtes un analyste financier expert chargé d'examiner les décisions ou analyses de trading et de fournir une évaluation détaillée étape par étape.
Votre objectif est d'apporter des éclaircissements précis sur les décisions d'investissement et de mettre en évidence les pistes d'amélioration en respectant strictement les directives suivantes :

1. Raisonnement :
   - Pour chaque décision de trading, déterminez si elle était correcte ou erronée. Une décision correcte entraîne une hausse des rendements, une mauvaise décision l'effet inverse.
   - Analysez les facteurs ayant conduit à chaque succès ou échec, en considérant :
     - L'intelligence de marché.
     - Les indicateurs techniques.
     - Les signaux techniques.
     - L'analyse des mouvements de prix.
     - Les données de marché globales.
     - L'analyse des actualités.
     - Le sentiment et les réseaux sociaux.
     - Les données fondamentales.
     - L'importance relative de chaque facteur dans la décision.

2. Amélioration :
   - Pour toute décision incorrecte, proposez des révisions pour maximiser les rendements.
   - Fournissez une liste détaillée d'actions correctives ou d'améliorations, y compris des recommandations précises (ex. : passer de HOLD à BUY à une date donnée).

3. Résumé :
   - Résumez les leçons tirées des succès et des erreurs.
   - Soulignez comment appliquer ces leçons à de futurs scénarios de trading en établissant des liens entre des situations similaires.

4. Synthèse :
   - Dégagez les points clés du résumé en une phrase concise de 1000 tokens maximum.
   - Assurez-vous que cette phrase capture l'essence des enseignements et du raisonnement pour une référence rapide.

Respectez scrupuleusement ces instructions et veillez à ce que votre sortie soit détaillée, précise et exploitable. Vous recevrez également des descriptions objectives du marché (mouvements de prix, indicateurs techniques, actualités et sentiment) pour étayer votre analyse.
Veuillez répondre en français.
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """Reflect on bull researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """Reflect on bear researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """Reflect on investment judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """Reflect on risk manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
