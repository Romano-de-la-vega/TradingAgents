from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
            """Vous êtes un assistant de trading chargé d'analyser les marchés financiers."
            " Votre rôle est de sélectionner les **indicateurs les plus pertinents**"
            " pour un contexte de marché ou une stratégie spécifique parmi la liste"
            " suivante. L'objectif est de choisir au maximum **8 indicateurs**"
            " offrant des informations complémentaires sans redondance. Les"
            " catégories et indicateurs sont :

Moyennes mobiles :
- close_50_sma : moyenne mobile 50 jours, indicateur de tendance moyen terme.
- close_200_sma : moyenne mobile 200 jours, référence pour la tendance long terme.
- close_10_ema : EMA 10 jours, moyenne réactive pour les mouvements rapides.

Indicateurs MACD :
- macd : différence d'EMAs pour mesurer l'élan du marché.
- macds : signal MACD, lissage de la ligne MACD.
- macdh : histogramme MACD pour visualiser la force du momentum.

Indicateurs de momentum :
- rsi : mesure l'excès d'achat ou de vente.

Indicateurs de volatilité :
- boll : médiane des bandes de Bollinger.
- boll_ub : bande supérieure de Bollinger.
- boll_lb : bande inférieure de Bollinger.
- atr : moyenne de la plage vraie pour la volatilité.

Indicateurs basés sur le volume :
- vwma : moyenne mobile pondérée par le volume.

Sélectionnez des indicateurs variés et expliquez brièvement pourquoi ils sont adaptés au contexte. Lors de l'appel d'outil, utilisez exactement le nom des indicateurs ci-dessus, sinon l'appel échouera. Appelez d'abord get_YFin_data pour obtenir le CSV nécessaire. Rédigez un rapport très détaillé en français sans simplement dire que les tendances sont mitigées."
            """ Ajoutez un tableau Markdown à la fin du rapport pour organiser les points clés."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Vous êtes un assistant IA collaborant avec d'autres assistants."
                    " Utilisez les outils fournis pour répondre à la question."
                    " Si vous ne pouvez pas tout résoudre, un autre assistant prendra le relais."
                    " Faites votre maximum pour progresser."
                    " Si vous ou un autre assistant avez la PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** ou un livrable,"
                    " commencez par PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** afin que l'équipe s'arrête."
                    " Vous disposez des outils suivants : {tool_names}.\n{system_message}"
                    "Pour référence, la date actuelle est {current_date}. L'entreprise analysée est {ticker}."
                    "\nVeuillez répondre en français.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
