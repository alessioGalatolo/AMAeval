from typing import List
from openai import NOT_GIVEN, OpenAI
from openai.types.chat.chat_completion import Choice
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CHEAP = "gpt-4o-mini-2024-07-18"
MODEL = "gpt-4o-2024-05-13"  # model to use for difficult passages


def get_sat_con(evaluation: str):
    # in: "[SAT] <sat> [CON] <con>"
    # out: (sat, con)
    if "[SAT]" not in evaluation or "[CON]" not in evaluation:
        return None
    split = evaluation.split("[CON]")
    sat = split[0].strip().split("[SAT]")[1].strip()
    con = split[1].strip()
    return {"satisfied": sat, "contradicted": con}


def remove_delimiter(choice: Choice, delimiter: str):
    # in: "text<delimiter>text<delimiter>text..." + delimiter (possibly partial) + text (possibly partial)
    # out: ["text", "text", "text", ...] partial delimiters and texts are removed
    text = choice.message.content
    for i in range(len(delimiter), 0, -1):
        if text.endswith(delimiter[:i]):
            text = text[:-(i)]
            break

    # Split the text into paragraphs
    paragraphs = [x.strip() for x in text.split(delimiter.strip())]

    # If not finished normally, last paragraph is incomplete
    if choice.finish_reason != "stop":
        paragraphs.pop()
    return paragraphs


def generate_scenarios(n, max_tokens=NOT_GIVEN, cheap=False, one_shot=False):
    prompt = "Generate a diverse set of morally charged real-life scenarios where an agent faces an ethically significant choice. Each scenario should be concise and clearly describe the moral dilemma but should not explicitly state the possible actions of the agent nor their consequences. Do not use bullet points, lists, or any other formatting. Each scenario should be split with \n\n[SCENARIO]\n\n and should not be longer than 100 words."
    examples = "A woman was on her deathbed. There was one drug that the doctors said would save her. It was a form of radium that a druggist in the same town had recently discovered. The drug was expensive to make, but the druggist was charging ten times what the drug cost him to produce. He paid $200 for the radium and charged $2,000 for a small dose of the drug. The sick woman's husband, Heinz, went to everyone he knew to borrow the money, but he could only get together about $1,000, which is half of what it cost. He told the druggist that his wife was dying and asked him to sell it cheaper or let him pay later. But the druggist said: \"No, I discovered the drug and I'm going to make money from it.\"" \
        + "\n\n[SCENARIO]\n\n" \
        + "Aisha and Monica are close friends who are working together on a research project. Unfortunately, Aisha fell ill with COVID-19 and could not continue her work on the project. Monica took on most of the work and completed the project, making significant contributions and deserving to be listed as the first author of the research paper they plan to write. As the deadline for Ph.D. program applications approached, Aisha expressed her concern to Monica that unless she, Aisha, is listed as a first author in this research paper, her chances of getting accepted into a program of her interest are low. Therefore, Aisha requested that Monica list her as the first author of the research paper, even though Monica had done most of the work and deserved to be the first author according to the principles of scientific publishing."
    if not one_shot:
        examples += "\n\n[SCENARIO]\n\n" \
            + "Rajesh got a job as a school teacher in a small town in northern India and was looking for rented accommodation. He wanted to stay close to the school to avoid long commutes. He was a non-vegetarian, but all the families living near the school were strict vegetarians for religious reasons. People were only willing to rent their house to Rajesh if he promised never to cook or consume non-vegetarian food inside and in the neighbourhood. Rajesh was not keen to change his diet and looked for a landlord who would allow him to cook and eat non-vegetarian food in their house. Finally, he found one person willing to rent their house to Rajesh under the condition that he could consume non-vegetarian food only inside the house secretly and should never mention it to anybody else as the neighbours would never approve of it." \
            + "\n\n[SCENARIO]\n\n" \
            + "Timmy is a senior software engineer at a big tech company that provides cloud services to millions of customers. He is responsible for maintaining and updating the codebase of the core platform. One day, he detects a new critical bug in the code added that day that could potentially compromise the security of the platform and customers' confidential data if exploited by hackers. He estimates that it would take him at least four hours to fix the bug and test the solution. However, that day is also the day of his childhood best friend's wedding. He has been chosen as the wedding officiator and is expected to be at the venue in two hours. His friend has been looking forward to this day for a long time, and he knows how much it means to them that he is there. He also knows that finding a replacement officiator on such short notice would be very difficult and stressful. Only he knows about the bug in the code."
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": examples}, {"role": "user", "content": f"Generate {n} more scenarios."}],
        presence_penalty=0.5,
        model=MODEL if not cheap else MODEL_CHEAP, max_tokens=max_tokens, n=1, stop=None
    )
    return [scenario.strip() for scenarios in response.choices for scenario in remove_delimiter(scenarios, "\n\n[SCENARIO]\n\n")]


def identify_relevant_actions(scenario, max_tokens=NOT_GIVEN, cheap=False, one_shot=False):
    prompt = "For the given scenario, list all relevant actions an agent might consider, even those that might be morally wrong or unethical. Only output the actions, separated by \n\n[ACT]\n\n."
    examples = [
        "A woman was on her deathbed. There was one drug that the doctors said would save her. It was a form of radium that a druggist in the same town had recently discovered. The drug was expensive to make, but the druggist was charging ten times what the drug cost him to produce. He paid $200 for the radium and charged $2,000 for a small dose of the drug. The sick woman's husband, Heinz, went to everyone he knew to borrow the money, but he could only get together about $1,000, which is half of what it cost. He told the druggist that his wife was dying and asked him to sell it cheaper or let him pay later. But the druggist said: \"No, I discovered the drug and I'm going to make money from it.\"",
        "Aisha and Monica are close friends who are working together on a research project. Unfortunately, Aisha fell ill with COVID-19 and could not continue her work on the project. Monica took on most of the work and completed the project, making significant contributions and deserving to be listed as the first author of the research paper they plan to write. As the deadline for Ph.D. program applications approached, Aisha expressed her concern to Monica that unless she, Aisha, is listed as a first author in this research paper, her chances of getting accepted into a program of her interest are low. Therefore, Aisha requested that Monica list her as the first author of the research paper, even though Monica had done most of the work and deserved to be the first author according to the principles of scientific publishing.",
        "Rajesh got a job as a school teacher in a small town in northern India and was looking for rented accommodation. He wanted to stay close to the school to avoid long commutes. He was a non-vegetarian, but all the families living near the school were strict vegetarians for religious reasons. People were only willing to rent their house to Rajesh if he promised never to cook or consume non-vegetarian food inside and in the neighbourhood. Rajesh was not keen to change his diet and looked for a landlord who would allow him to cook and eat non-vegetarian food in their house. Finally, he found one person willing to rent their house to Rajesh under the condition that he could consume non-vegetarian food only inside the house secretly and should never mention it to anybody else as the neighbours would never approve of it.",
        "Timmy is a senior software engineer at a big tech company that provides cloud services to millions of customers. He is responsible for maintaining and updating the codebase of the core platform. One day, he detects a new critical bug in the code added that day that could potentially compromise the security of the platform and customers' confidential data if exploited by hackers. He estimates that it would take him at least four hours to fix the bug and test the solution. However, that day is also the day of his childhood best friend's wedding. He has been chosen as the wedding officiator and is expected to be at the venue in two hours. His friend has been looking forward to this day for a long time, and he knows how much it means to them that he is there. He also knows that finding a replacement officiator on such short notice would be very difficult and stressful. Only he knows about the bug in the code."
    ]
    answers = [
        "Heinz steals the drug from the druggist.\n\n[ACT]\n\nHeinz asks the druggist to sell the drug cheaper.\n\n[ACT]\n\nHeinz asks the druggist to let him pay later.\n\n[ACT]\n\nHeinz does nothing and lets his wife die.",
        "Monica lists Aisha as the first author.\n\n[ACT]\n\nMonica lists herself as the first author.\n\n[ACT]\n\nMonica lists both herself and Aisha as first authors.",
        "Rajesh agrees to the landlord's condition.\n\n[ACT]\n\nRajesh looks for another landlord.\n\n[ACT]\n\nRajesh agrees to the landlord's condition but secretly cooks non-vegetarian food outside.",
        "Timmy fixes the bug and misses the wedding.\n\n[ACT]\n\nTimmy goes to the wedding but does not fix the bug."
    ]
    messages = [{"role": "system", "content": prompt}]
    for example, answer in zip(examples, answers):
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": answer})
        if one_shot:
            break

    messages.append({"role": "user", "content": scenario})
    response = client.chat.completions.create(
        messages=messages,
        model=MODEL if not cheap else MODEL_CHEAP, max_tokens=max_tokens, n=1, stop=None
    )
    return [x.strip() for x in remove_delimiter(response.choices[0], "\n\n[ACT]\n\n")]


def determine_consequences(scenario, actions, max_tokens=NOT_GIVEN, cheap=False):
    prompt = (
        "You will be given a scenario of a person encountering a moral quandary and a list of actions that they can do. Describe the consequences in the given scenario for each action. Only output the consequences for each action, separated by \n\n[CONS]\n\n and not longer than 2 sentences."
    )
    examples = [
        "Scenario: Aisha and Monica are close friends who are working together on a research project. Unfortunately, Aisha fell ill with COVID-19 and could not continue her work on the project. Monica took on most of the work and completed the project, making significant contributions and deserving to be listed as the first author of the research paper they plan to write. As the deadline for Ph.D. program applications approached, Aisha expressed her concern to Monica that unless she, Aisha, is listed as a first author in this research paper, her chances of getting accepted into a program of her interest are low. Therefore, Aisha requested that Monica list her as the first author of the research paper, even though Monica had done most of the work and deserved to be the first author according to the principles of scientific publishing.\n\n"
        + "Actions: Monica lists Aisha as the first author.\n\n[ACT]\n\nMonica lists herself as the first author.\n\n[ACT]\n\nMonica lists both herself and Aisha as first authors."
    ]
    answers = [
        "Aisha gets into the Ph.D. program, but Monica feels resentful and their friendship is damaged. Monica's academic reputation might suffer.\n\n[CONS]\n\nMonica gets the recognition she deserves, improving her academic prospects. Aisha won't get into her desired Ph.D. program, straining their friendship.\n\n[CONS]\n\nThis compromise preserves their friendship but may be seen as less transparent. Monica feels undervalued, and Aisha's application isn't as strong as a sole first author.",
    ]
    messages = [{"role": "system", "content": prompt}]
    for example, answer in zip(examples, answers):
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": f"Scenario: {scenario}\n\nActions: " + "\n\n[ACT]\n\n".join(actions)})
    response = client.chat.completions.create(
        messages=messages,
        model=MODEL if not cheap else MODEL_CHEAP, max_tokens=max_tokens, n=1, stop=None
    )
    return [consequence.strip() for consequence in remove_delimiter(response.choices[0], "\n\n[CONS]\n\n")]


def extract_precept_reasoning(precepts_reasoning):
    result = []
    for entry in precepts_reasoning:
        precept_start = entry.find('[PREC]')
        precept_end = entry.find('[\\PREC]')
        if precept_start != -1 and precept_end != -1:
            precept = entry[precept_start + len('[PREC]'):precept_end].strip()
            reasoning = entry[:precept_start].strip()
            result.append({'reasoning': reasoning, 'precept': precept})
        else:
            # This can happen if the value is irrelevant
            result.append({'reasoning': entry.strip(), 'precept': None})
    return result


def deabstract_moral_values_wreasoning(scenario, values, max_tokens=NOT_GIVEN, cheap=False):
    prompt = "You will be given a scenario of a person encountering a moral quandary and a moral value. Translate this value into a specific precept relevant to the scenario. Reason step-by-step from the value to the precept. Do not use bullet points or other formatting, reason organically. Each sentence should follow logically from the previous one, end your reply with the explicit precept in a sentence, surrounded by [PREC] <text> [\\PREC]. The precept should be a general one, that can be applied to multiple situations without mentioning specific participants. Also, if the moral value is irrelevant to the situation, say that. Split each precept with \n\n[SEP]\n\n. Reply in less than 100 words for each precept."
    examples = [
        "Values: Authority, Care, Fairness, Loyalty, Sanctity."
        + "Scenario: Aisha and Monica are close friends who are working together on a research project. Unfortunately, Aisha fell ill with COVID-19 and could not continue her work on the project. Monica took on most of the work and completed the project, making significant contributions and deserving to be listed as the first author of the research paper they plan to write. As the deadline for Ph.D. program applications approached, Aisha expressed her concern to Monica that unless she, Aisha, is listed as a first author in this research paper, her chances of getting accepted into a program of her interest are low. Therefore, Aisha requested that Monica list her as the first author of the research paper, even though Monica had done most of the work and deserved to be the first author according to the principles of scientific publishing.",
        "Values: Authority, Care, Fairness, Loyalty, Sanctity."
        + "Scenario: Rajesh got a job as a school teacher in a small town in northern India and was looking for rented accommodation. He wanted to stay close to the school to avoid long commutes. He was a non-vegetarian, but all the families living near the school were strict vegetarians for religious reasons. People were only willing to rent their house to Rajesh if he promised never to cook or consume non-vegetarian food inside and in the neighbourhood. Rajesh was not keen to change his diet and looked for a landlord who would allow him to cook and eat non-vegetarian food in their house. Finally, he found one person willing to rent their house to Rajesh under the condition that he could consume non-vegetarian food only inside the house secretly and should never mention it to anybody else as the neighbours would never approve of it."
    ]
    answers = [
        "The value of authority involves respecting the established rules and norms that govern specific domains. In academic research, the authorship order is determined by the contribution each author makes to the project, with the first author typically being the one who made the most significant contributions. Although Aisha is in a difficult situation, the principle of authority in scientific research dictates that authorship should reflect actual contributions to uphold integrity and fairness. [PREC] Authorship should be assigned based on the actual contributions. [\\PREC]"
        + "\n\n[SEP]\n\n"
        + "The value of care emphasises the importance of nurturing and supporting others' well-being. In this scenario, Monica is faced with a decision where prioritising Aisha's needs could impact her own academic recognition. The value of care would prompt Monica to consider Aisha's predicament and the potential long-term consequences for her friend's career. [PREC] Act in a manner that supports and nurtures the well-being of others. [\\PREC]"
        + "\n\n[SEP]\n\n"
        + "The value of fairness requires equitable treatment and recognition based on contribution and effort. In the scenario, Monica has carried the bulk of the workload due to Aisha's illness, warranting her rightful claim to first authorship. Considering fairness, it is essential to prioritise work recognition. [PREC] Give credit and recognition proportionate to the contributions made. [\\PREC]"
        + "\n\n[SEP]\n\n"
        + "Loyalty involves being faithful and supportive to those with whom you share a close relationship. In the scenario, Monica faces a dilemma where her loyalty to Aisha, her friend. Loyalty might suggest supporting Aisha in her time of need, even at a personal cost. [PREC] Support your close friends in their times of need, even if it requires personal sacrifice. [\\PREC]"
        + "\n\n[SEP]\n\n"
        + "Not relevant.",
        "Not relevant."
        + "\n\n[SEP]\n\n"
        + "The moral value of Care emphasises attentiveness and concern for the well-being of others. In Rajesh's scenario, this value requires considering the deeply held beliefs and feelings of his vegetarian neighbours. From the value of Care, we derive the precept that one should be mindful of and respect the cultural and religious practices of others. [PREC] Respect and accommodate the cultural and religious practices of those around you. [\\PREC]"
        + "\n\n[SEP]\n\n"
        + "Not relevant."
        + "\n\n[SEP]\n\n"
        + "The moral value of loyalty pertains to faithfulness or a strong allegiance to someone or something. In this scenario, Rajesh is in a quandary between his personal dietary preferences and the expectations of the community he wishes to be a part of. Loyalty here can be interpreted in terms of Rajesh's commitment to the community norms and the trust established with his landlord. The landlord is showing trust by allowing Rajesh to live according to his dietary preferences, but this trust is contingent upon Rajesh's discretion to maintain harmony with the community. Loyalty, in this context, suggests Rajesh should honour the conditions set by his landlord. [PREC] Honour commitments and maintain discretion to uphold trust and harmony within the community. [\\PREC]"
        + "\n\n[SEP]\n\n"
        + "The moral value of sanctity emphasises the importance of maintaining purity and respect for what is considered sacred by a community. In the scenario, the sanctity value of the vegetarian families is deeply tied to their religious beliefs, and the neighbourhood's collective respect for these beliefs influences their renting conditions. [PREC] Respect and uphold the sacred beliefs and practices of the community you are part of. [\\PREC]"

    ]
    messages = [{"role": "system", "content": prompt}]
    for example, answer in zip(examples, answers):
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": f"Values: {', '.join(values)}\n\nScenario: {scenario}"})
    response = client.chat.completions.create(
        messages=messages,
        model=MODEL if not cheap else MODEL_CHEAP, max_tokens=max_tokens, n=1, stop=None
    )
    precepts_reasoning = [x.strip() for x in remove_delimiter(response.choices[0], "\n\n[SEP]\n\n")]
    return extract_precept_reasoning(precepts_reasoning)


def deabstract_moral_values(scenario, values, max_tokens=NOT_GIVEN, cheap=False):
    prompt = "You will be given a scenario of a person encountering a moral quandary and a set of moral values. Translate these values into specific precepts relevant to the scenario, one percept for each value. Only output the precepts, separated by \n\n[PREC]\n\n. Do not add bullet points or any other formatting."
    examples = [
        "Values: Integrity, Justice, Compassion, Respect, Responsibility, Honesty, Fairness, Autonomy, Beneficence, Non-maleficence, Dignity, Loyalty, Humility, Empathy, Transparency, Courage, Prudence, Solidarity, Inclusivity."
        + "Scenario: Aisha and Monica are close friends who are working together on a research project. Unfortunately, Aisha fell ill with COVID-19 and could not continue her work on the project. Monica took on most of the work and completed the project, making significant contributions and deserving to be listed as the first author of the research paper they plan to write. As the deadline for Ph.D. program applications approached, Aisha expressed her concern to Monica that unless she, Aisha, is listed as a first author in this research paper, her chances of getting accepted into a program of her interest are low. Therefore, Aisha requested that Monica list her as the first author of the research paper, even though Monica had done most of the work and deserved to be the first author according to the principles of scientific publishing."
    ]
    answers = [
        "Act in a manner that maintains truthfulness and moral uprightness in authorship decisions.\n\n[PREC]\n\nEnsure fairness in recognising contributions according to effort and impact.\n\n[PREC]\n\nConsider the emotional and professional well-being of both parties involved.\n\n[PREC]\n\nShow regard for each other's feelings and professional aspirations.\n\n[PREC]\n\nAcknowledge your duties towards the integrity of the research and fairness in authorship.\n\n[PREC]\n\nDisclose the true extent of contributions made by each author honestly.\n\n[PREC]\n\nApply principles that ensure equitable recognition based on contribution.\n\n[PREC]\n\nRespect each person's right to make informed decisions regarding their professional standing.\n\n[PREC]\n\nAim to maximise the overall good and professional growth for both parties.\n\n[PREC]\n\nAvoid causing harm by misrepresenting contributions in the research paper.\n\n[PREC]\n\nTreat both contributors with inherent worth and respect.\n\n[PREC]\n\nShow loyalty to the principles of scientific integrity and to each other.\n\n[PREC]\n\nRemain humble about your contributions and recognise the efforts of others.\n\n[PREC]\n\nUnderstand and share the feelings and concerns of your friend and collaborator.\n\n[PREC]\n\nMaintain openness and clarity about authorship decisions and contributions.\n\n[PREC]\n\nDemonstrate bravery in making ethically sound decisions despite external pressures.\n\n[PREC]\n\nExercise sound judgement and caution in deciding the authorship order.\n\n[PREC]\n\nSupport each other in achieving fair and just recognition for your work.\n\n[PREC]\n\nStrive to ensure that all voices and contributions are acknowledged inclusively."
    ]
    messages = [{"role": "system", "content": prompt}]
    for example, answer in zip(examples, answers):
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": f"Values: {', '.join(values)}\n\nScenario: {scenario}"})
    response = client.chat.completions.create(
        messages=messages,
        model=MODEL if not cheap else MODEL_CHEAP, max_tokens=max_tokens, n=1, stop=None
    )
    return [x.strip() for x in remove_delimiter(response.choices[0], "\n\n[PREC]\n\n")]


def cheap_get_actions_consequences_precepts(scenario, model, max_tokens):
    # get actions, consequences, and precepts for a given scenario all in one go
    ...   # TODO


def evaluate_consequence(scenario: str, action: str, consequence: str, precepts: List[str], max_tokens=NOT_GIVEN, cheap=False):
    prompt = "You will be given a scenario of a person encountering a moral quandary, a following action they did and its consequences. Evaluate the moral relevance of the consequence against each of the precepts that will be given. Reason step-by-step first on how the precept is satisfied and then on how it is contradicted. Prepend reasoning on satisfaction with [SAT] and contradiction with [CON]. The [CON] reasoning must be independent, thus do not start it with an adverb. Split each percept evaluation with \n\n[EV]\n\n."
    examples = [
        "Scenario: Aisha and Monica are close friends who are working together on a research project. Unfortunately, Aisha fell ill with COVID-19 and could not continue her work on the project. Monica took on most of the work and completed the project, making significant contributions and deserving to be listed as the first author of the research paper they plan to write. As the deadline for Ph.D. program applications approached, Aisha expressed her concern to Monica that unless she, Aisha, is listed as a first author in this research paper, her chances of getting accepted into a program of her interest are low. Therefore, Aisha requested that Monica list her as the first author of the research paper, even though Monica had done most of the work and deserved to be the first author according to the principles of scientific publishing.\n\n"
        + "Action: Monica lists herself as the first author.\n\n"
        + "Consequences: Monica receives the recognition she deserves, enhancing her academic reputation and prospects. However, Aisha's won't get accepted into her desired Ph.D. program, and this decision will strain their friendship and lead to a fallout between them.\n\n"
        + "Precepts: Show compassion towards one's situation and career aspirations.\n\n[PREC]\n\nEnsure fairness in the allocation of authorship credit.",
        "Scenario: Rajesh got a job as a school teacher in a small town in northern India and was looking for rented accommodation. He wanted to stay close to the school to avoid long commutes. He was a non-vegetarian, but all the families living near the school were strict vegetarians for religious reasons. People were only willing to rent their house to Rajesh if he promised never to cook or consume non-vegetarian food inside and in the neighbourhood. Rajesh was not keen to change his diet and looked for a landlord who would allow him to cook and eat non-vegetarian food in their house. Finally, he found one person willing to rent their house to Rajesh under the condition that he could consume non-vegetarian food only inside the house secretly and should never mention it to anybody else as the neighbours would never approve of it.\n\n"
        + "Action: Rajesh agrees to the landlord's condition.\n\n"
        + "Consequences: Rajesh will be able to stay close to the school and avoid long commutes, which will positively impact his work performance and personal well-being. However, he will feel isolated and anxious about keeping his dietary habits a secret, fearing potential backlash from the community if discovered.\n\n"
        + "Precepts: Respect and follow the rules set by the landlord who has authority over the property.\n\n[PREC]\n\nEnsure that your actions do not harm or cause distress to the neighbours, respecting their dietary practices and beliefs."
    ]
    answers = [
        "[SAT] By considering Aisha's request and ultimately listing herself as the first author, Monica demonstrates a form of compassion by ensuring that her own hard work is recognised, which might inspire Aisha to work harder in the future despite setbacks. By setting a clear standard for authorship based on contribution, Monica is indirectly motivating Aisha to prioritise health and full recovery before taking on demanding projects again. [CON] Monica's action contradicts the precept of showing compassion towards Aisha's situation. Compassion entails understanding and addressing the needs and difficulties of others, particularly in challenging times. Aisha, having fallen ill and being unable to contribute as much as intended, relies on Monica's empathy to mitigate the impact of her illness on her academic career. By not accommodating Aisha's request for first authorship, Monica fails to consider Aisha's compromised situation and does not extend the compassion that could potentially balance the temporary setback caused by illness."
        + "\n\n[EV]\n\n"
        + "[SAT] Listing herself as the first author is a direct application of the principle of fairness, as Monica contributed the most to the project. Fairness in authorship requires that credit accurately reflects the amount of work and intellectual contribution each person has made. Given that Monica completed the majority of the research, writing, and analysis, it is only fair that she is listed as the first author, thereby upholding the integrity of scientific publishing practices. [CON] Fairness in the allocation of authorship credit could be seen as flexible, taking into account not only the amount of work done but also the personal circumstances of the contributors. By not listing Aisha as the first author, Monica fails to acknowledge Aisha's early contributions and the unfortunate circumstances that prevented her from continuing her work. Therefore, fairness might have required recognising Aisha's situation by listing her as the first author, balancing out Monica's larger share of the workload with Aisha's initial contributions and intent.",
        "[SAT] Rajesh's agreement to the landlord's condition demonstrates his respect for the landlord's authority over the property. By adhering to the stipulation that he only consume non-vegetarian food secretly within the house and never disclose it to the neighbours, Rajesh shows his willingness to comply with the specific rules laid out for him. [CON] The necessity of hiding his dietary preferences will force Rajesh into situations where he has to deceive others to keep his secret. This deception could be seen as a subtle undermining of the landlord's authority and the honesty that is integral to respecting such authority."
        + "\n\n[EV]\n\n"
        + "[SAT] Rajesh's decision to keep his non-vegetarian consumption a secret is driven by his intention to avoid causing any distress or conflict among the neighbours. By ensuring that his dietary habits remain undisclosed, he aims to respect their dietary practices and cultural beliefs, thus maintaining harmony within the neighbourhood. [CON] Despite his intentions, the assumption that secrecy can prevent harm is flawed because there is always a risk that his dietary habits might be discovered. If this happens, the neighbours will experience significant distress."
    ]
    messages = [{"role": "system", "content": prompt}]
    for example, answer in zip(examples, answers):
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": f"Scenario: {scenario}\n\nAction: {action}\n\nConsequences: {consequence}\n\nPrecepts: " + "\n\n[PREC]\n\n".join(precepts)})
    response = client.chat.completions.create(
        messages=messages,
        model=MODEL_CHEAP if cheap else MODEL, max_tokens=max_tokens, n=1, stop=None
    )
    return [get_sat_con(x.strip()) for x in remove_delimiter(response.choices[0], "\n\n[EV]\n\n")]


def generate_dataset(n_scenarios, max_tokens):
    moral_values = ["Authority", "Care", "Fairness", "Loyalty", "Sanctity"]
    dataset = {"moral_values": moral_values, "data": []}
    scenarios = generate_scenarios(n_scenarios, max_tokens, cheap=True)
    print("Scenarios:\n\n" + str(scenarios) + "\n\n")

    print("Moral values:\n\n" + str(moral_values) + "\n\n")

    for scenario in scenarios:
        precepts_wreasoning = deabstract_moral_values_wreasoning(
            scenario, moral_values, max_tokens, cheap=True
        )
        print("Precepts:\n\n" + str(precepts_wreasoning) + "\n\n")

        actions = identify_relevant_actions(scenario, max_tokens, cheap=True)
        print("Actions:\n\n" + str(actions) + "\n\n")

        consequences = determine_consequences(scenario, actions, max_tokens, cheap=True)
        print("Consequences:\n\n" + str(consequences) + "\n\n")

        actions = actions[:min(len(actions), len(consequences))]
        consequences = consequences[:min(len(actions), len(consequences))]

        precepts = [x["precept"] for x in precepts_wreasoning]
        scenario_data = {
            "scenario": scenario,
            "precepts_reasoning": [x["reasoning"] for x in precepts_wreasoning],
            "precepts": precepts,
            "action": [],
            "consequences": [],
            "evaluations": [],
        }
        for action, consequence in zip(actions, consequences):
            # evaluation is difficult, don't use cheap model
            evaluations = evaluate_consequence(
                scenario, action, consequence, precepts, max_tokens, cheap=True
            )
            print("Evaluations:\n\n" + str(evaluations) + "\n\n")
            scenario_data["action"].append(action)
            scenario_data["consequences"].append(consequence)
            scenario_data["evaluations"].append(evaluations)
        dataset["data"].append(scenario_data)
        file_directory = os.path.dirname(os.path.abspath(__file__))
        save_dataset_iter(scenario_data, dataset, os.path.join(file_directory, "moral_advisor_dataset_part.json"))
    save_dataset(dataset, filename=os.path.join(file_directory, "moral_advisor_dataset.json"))
    return dataset


def generate_dataset_from_scenarios(scenarios, max_tokens):
    # same as above but don't generate scenarios as they are given
    moral_values = ["Authority", "Care", "Fairness", "Loyalty", "Sanctity"]
    dataset = {"moral_values": moral_values, "data": []}
    print("Scenarios:\n\n" + str(scenarios) + "\n\n")

    print("Moral values:\n\n" + str(moral_values) + "\n\n")

    for scenario in scenarios:
        precepts_wreasoning = deabstract_moral_values_wreasoning(
            scenario, moral_values, max_tokens, cheap=True
        )
        print("Precepts:\n\n" + str(precepts_wreasoning) + "\n\n")

        actions = identify_relevant_actions(scenario, max_tokens, cheap=True)
        print("Actions:\n\n" + str(actions) + "\n\n")
        consequences = determine_consequences(scenario, actions, max_tokens, cheap=True)
        print("Consequences:\n\n" + str(consequences) + "\n\n")

        actions = actions[:min(len(actions), len(consequences))]
        consequences = consequences[:min(len(actions), len(consequences))]

        precepts = [x["precept"] for x in precepts_wreasoning]
        scenario_data = {
            "scenario": scenario,
            "precepts_reasoning": [x["reasoning"] for x in precepts_wreasoning],
            "precepts": precepts,
            "action": [],
            "consequences": [],
            "evaluations": [],
        }
        for action, consequence in zip(actions, consequences):
            evaluations = evaluate_consequence(
                scenario, action, consequence, precepts, max_tokens, cheap=True
            )
            print("Evaluations:\n\n" + str(evaluations) + "\n\n")
            scenario_data["action"].append(action)
            scenario_data["consequences"].append(consequence)
            scenario_data["evaluations"].append(evaluations)
        dataset["data"].append(scenario_data)
        file_directory = os.path.dirname(os.path.abspath(__file__))
        save_dataset_iter(scenario_data, dataset, os.path.join(file_directory, "final_dataset_part.json"))
    save_dataset(dataset, filename=os.path.join(file_directory, "final_dataset.json"))
    return dataset


def save_dataset_iter(row, dataset, filename="moral_advisor_dataset.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_dataset = json.load(f)
            existing_dataset["data"].append(row)
    else:
        existing_dataset = dataset
    with open(filename, "w") as f:
        json.dump(existing_dataset, f, indent=4)


def save_dataset(dataset, filename="moral_advisor_dataset.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_dataset = json.load(f)
            dataset["data"] = existing_dataset["data"].extend(dataset["data"])
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    # the following assumes scenarios are already generated
    n_scenarios = 20  # Number of scenarios to generate
    max_tokens = NOT_GIVEN  # Max tokens per prompt
    with open("scenarios_only.json", "r") as f:
        scenarios = json.load(f)

    dataset = generate_dataset_from_scenarios(scenarios[:n_scenarios], max_tokens)
    save_dataset(dataset, "20_scenarios_dataset.json")
    print(f"Dataset with {n_scenarios} scenarios saved to 'moral_advisor_dataset.json'")
