from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_stock_news_openai]
        else:
            tools = [
                toolkit.get_reddit_stock_info,
            ]

        system_message = (
            "Vous êtes un analyste spécialisé dans les réseaux sociaux et les"
            " actualités d'entreprise. Votre tâche est d'analyser les publications"
            " sur les réseaux sociaux, les nouvelles récentes de l'entreprise et le"
            " sentiment du public sur la dernière semaine. Votre objectif est de"
            " rédiger un rapport détaillé en français présentant votre analyse, vos"
            " idées et leurs implications pour les traders et investisseurs."
            " Consultez toutes les sources possibles (médias sociaux, sentiment,"
            " actualités). Ne vous contentez pas d'indiquer que les tendances sont"
            " mitigées : fournissez une analyse fine pouvant aider à la décision."
            """ Ajoutez un tableau Markdown à la fin du rapport pour résumer les"
            " points clés de façon lisible."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Vous êtes un assistant IA coopérant avec d'autres assistants."
                    " Utilisez les outils fournis pour répondre à la question."
                    " Si vous ne pouvez pas répondre entièrement, un autre assistant s'en chargera."
                    " Faites le maximum pour progresser."
                    " Si vous ou un autre assistant disposez de la PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** ou d'un livrable,"
                    " commencez la réponse par PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** afin que l'équipe s'arrête."
                    " Vous avez accès aux outils suivants : {tool_names}.\n{system_message}"
                    "Pour référence, la date actuelle est {current_date}. La société analysée est {ticker}."
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
