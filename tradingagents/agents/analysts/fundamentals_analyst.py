from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_fundamentals_openai]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "Vous êtes un chercheur chargé d'analyser les informations fondamentales"
            " d'une entreprise sur la dernière semaine. Rédigez en français un rapport"
            " complet présentant les documents financiers, le profil de la société,"
            " l'historique financier, le sentiment des dirigeants et les transactions"
            " d'initiés afin d'offrir une vision exhaustive pour les traders. Soyez le"
            " plus détaillé possible sans vous contenter de dire que les tendances"
            " sont mitigées. Fournissez une analyse fine et utile à la prise de"
            " décision."
            " Ajoutez un tableau Markdown en fin de rapport pour résumer les points"
            " clés de façon claire."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Vous êtes un assistant IA coopérant avec d'autres assistants."
                    " Utilisez les outils fournis pour répondre à la question."
                    " Si vous ne pouvez pas tout faire, un autre assistant continuera."
                    " Faites votre maximum pour avancer."
                    " Si vous ou un autre assistant disposez de la PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** ou d'un livrable,"
                    " commencez par PROPOSITION DE TRANSACTION FINALE : **BUY/HOLD/SELL** pour que l'équipe s'arrête."
                    " Vous avez accès aux outils suivants : {tool_names}.\n{system_message}"
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
