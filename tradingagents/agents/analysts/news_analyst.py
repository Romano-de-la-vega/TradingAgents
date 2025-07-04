from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_global_news_openai, toolkit.get_google_news]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            "Vous êtes un chercheur chargé d'analyser les actualités récentes et"
            " les tendances de la dernière semaine. Rédigez en français un rapport"
            " complet sur la situation mondiale pertinente pour le trading et la"
            " macroéconomie. Consultez notamment les sources EODHD et Finnhub pour"
            " être exhaustif. Ne vous contentez pas de dire que les tendances sont"
            " mitigées : fournissez une analyse approfondie et nuancée pouvant aider"
            " les traders à prendre des décisions."
            """ Ajoutez un tableau Markdown à la fin pour présenter clairement les"
            " points clés du rapport."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Vous êtes un assistant IA coopérant avec d'autres assistants."
                    " Utilisez les outils fournis pour répondre à la question."
                    " Si vous ne pouvez pas tout résoudre, un autre assistant poursuivra."
                    " Faites votre possible pour progresser."
                    " Si vous ou un autre assistant détenez la PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** ou un livrable,"
                    " commencez votre réponse par PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** afin que l'équipe s'arrête."
                    " Vous avez accès aux outils suivants : {tool_names}.\n{system_message}"
                    "Pour référence, la date actuelle est {current_date}. Nous analysons la société {ticker}."
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
            "news_report": report,
        }

    return news_analyst_node
