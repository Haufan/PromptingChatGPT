# Projekt - Von Wörterbüchern bis GPT
# WS 2023/2024
# Dietmar Benndorf


from alive_progress import alive_bar
from openai import OpenAI
import pandas
import re
import requests
import wikipedia


def get_word_content(search_words):
    """Looks up search word information on wikipedia and dwds and saves it in a dict."""

    data_search_words = dict()

    print('Retrieving Information from Wikipedia and DWDS ...\n')
    with alive_bar(len(search_words), force_tty=True) as bar:

        for word in search_words:
            data_wiki = get_wiki(word)
            data_dwds = get_dwds(word)

            data_wiki[1] = re.sub(r'\n', ' ', data_wiki[1])

            data_search_words[word] = {'wiki_def': data_wiki[0],
                                       'wiki_full': data_wiki[1],
                                       'dwds_def': data_dwds[0],
                                       'dwds_alt': data_dwds[1],
                                       'dwds_con': data_dwds[2]}

        bar()

    return data_search_words


def get_wiki(word):
    """Looks up search word on wikipedia.de. If there is an entry, it extracts the definition and full text."""

    wikipedia.set_lang("de")

    if check_wiki_entry(word):
        info = wikipedia.page(word)
        full_info = info.content
        only_def = info.content[:info.content.index('\n')]
        return [only_def, full_info]
    else:
        return ['no entry', 'no entry']


def check_wiki_entry(word):
    """Checks if a word has a wikipedia entry. Returns True, if so and False if not."""

    url = f"https://en.wikipedia.org/w/api.php"
    params = {"action": "query",
              "list": "search",
              "srsearch": word,
              "format": "json"}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['query']['search']:
            return True
        else:
            return False
    except requests.RequestException as e:
        return False


def get_dwds(word):
    """Looks up search word on dwds.de. If there is an entry, it extracts the definition and alternatives."""

    list_word_alternativ = []
    _temp_word_definition = []
    _temp_word_context = []

    # Erstellt DWDS-URL
    word_url = re.sub(r' ', '%20', word)
    dwds_url = "https://www.dwds.de/wb/" + word_url

    # Checkt, ob URL exisitiert, wenn ja ruft sie auf
    if check_url(dwds_url):
        response = requests.get(dwds_url)
        content = response.text
        content = re.sub(r'\n', ' ', content)

        # Auslesen der Defintionen als String in einer Liste
        _definitions = re.findall(r'"@type" : "DefinedTerm".*?"description".*?}', content)

        for definition in _definitions:
            _temp_word_definition.append(re.findall(r'"description" : ".*?"', definition)[0])

        list_word_definition = [x[17:-1] for x in _temp_word_definition]

        # Auslesen der Verweise als String in einer Liste
        word_alternative = re.findall('class="dwdswb-verweis".*?&lt;/span', content)

        for alt in word_alternative:
            _temp_alt = re.findall('&gt;.*&lt;/span', alt)
            list_word_alternativ.append(_temp_alt[0][4:-9])

        # Auslesen der Belege als String in einer Liste
        _quotes = re.findall(r'"@type" : "Quotation".*?"text".*?}', content)

        for quote in _quotes:
            _temp_word_context.append(re.findall(r'"text" : ".*?"', quote)[0])

        list_word_context = [x[10:-1] for x in _temp_word_context]

        return [list_word_definition, list_word_alternativ, list_word_context]

    else:
        return ['no entry', 'no entry', 'no entry']


def check_url(url):
    """Checks if an url exist. Returns True for an existing url and False if it does not."""

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.RequestException:
        return False


def get_chatgpt(prompt, role):
    """Takes the content of the role and prompt and returns the ChatGPT message."""

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system",
                   "content": role},
                  {"role": "user", "content": prompt}])

    _temp_text = str(completion.choices[0].message)
    _temp_text = re.findall(r"content=(.*)role", _temp_text)

    return _temp_text[0][1:-3]


def main_prompting(search_words, examples, roles, data):
    """Retrieves Chat GPT reaction to different prompting options."""

    _temp_role = ''
    all_data_df = pandas.DataFrame(columns=['Word',
                                            'Wiki_def',
                                            'DWDS_def',
                                            'Zero',
                                            'Zero_dwds',
                                            'Zero_wiki',
                                            'Zero_both',
                                            'Few',
                                            'Few_dwds',
                                            'Few_wiki',
                                            'Few_both',
                                            'CoT',
                                            'CoT_dwds',
                                            'CoT_wiki',
                                            'CoT_both',
                                            'RAG_dwds',
                                            'RAG_wiki',
                                            'RAG_both'])

    print('\nRetrieving ChatGPT responses ...')
    with alive_bar(len(search_words), force_tty=True) as bar:

        for word in search_words:
            _temp_word_data = [word, data[word]['wiki_def'], data[word]['dwds_def']]

            # zero-shot prompting
            prompt = f'Definiere das folgende Wort: {word}.'
            _temp_word_data.append(get_chatgpt(prompt, roles[0]))

                # zero-shot prompting + DWDS Belege
            prompt = f'Definiere das folgende Wort: {word}. Nutze die folgenden Belege als Hilfe. ' \
                     f'Belege = {data[word]["dwds_con"]}'
            _temp_word_data.append(get_chatgpt(prompt, roles[0]))

            if data[word]['wiki_full'] != 'no entry':
                # zero-shot prompting + wikipedia article
                prompt = f'Definiere das folgende Wort: {word}. Nutze den folgenden Text als Hilfe. ' \
                         f'Text = {data[word]["wiki_full"]}'
                _temp_word_data.append(get_chatgpt(prompt, roles[0]))

                # zero-shot prompting + beide
                prompt = f'Definiere das folgende Wort: {word}. Nutze den folgenden Text und ' \
                         f'die folgenden Belege als Hilfe. ' \
                         f'Text = {data[word]["wiki_full"]}, Belege = {data[word]["dwds_con"]}'
                _temp_word_data.append(get_chatgpt(prompt, roles[0]))
            else:
                _temp_word_data.append('Kein Wiki-Eintrag')
                _temp_word_data.append('Kein Wiki-Eintrag')

            # few-shot-prompting
            for example in examples:
                _temp_role = _temp_role + \
                           f'Die Definition von {example} ist {examples[example]}'
            role = _temp_role
            prompt = f'Die Defintion von {word} ist ...'
            _temp_word_data.append(get_chatgpt(prompt, role))

                # few-shot-prompting + DWDS Belege
            prompt = f'Die Defintion von {word} ist ... Nutze die folgenden Belege als Hilfe. ' \
                     f'Belege = {data[word]["dwds_con"]}'
            _temp_word_data.append(get_chatgpt(prompt, role))

            if data[word]['wiki_full'] != 'no entry':
                # few-shot-prompting + wikipedia article
                prompt = f'Die Defintion von {word} ist ... Nutze den folgenden Text als Hilfe. ' \
                         f'Text = {data[word]["wiki_full"]}'
                _temp_word_data.append(get_chatgpt(prompt, role))

                # few-shot-prompting + beide
                prompt = f'Die Defintion von {word} ist ... Nutze den folgenden Text und ' \
                         f'die folgenden Belege als Hilfe. ' \
                         f'Text = {data[word]["wiki_full"]}, Belege = {data[word]["dwds_con"]}'
                _temp_word_data.append(get_chatgpt(prompt, role))
            else:
                _temp_word_data.append('Kein Wiki-Eintrag')
                _temp_word_data.append('Kein Wiki-Eintrag')

            # chain of thought
            prompt = f'Was ist die Definition von {word}? Erklär deine Gedankenschritte.'
            _temp_word_data.append(get_chatgpt(prompt, role))

                # chain of thought + DWDS Belege
            prompt = f'Was ist die Definition von {word}? Nutze die folgenden Belege als Hilfe und ' \
                     f'erklär deine Gedankenschritte. ' \
                     f'Belege = {data[word]["dwds_con"]}'
            _temp_word_data.append(get_chatgpt(prompt, role))

            if data[word]['wiki_full'] != 'no entry':
                # chain of thought + wikipedia article
                prompt = f'Was ist die Definition von {word}? Nutze den folgenden Text als Hilfe und ' \
                         f'erklär deine Gedankenschritte. ' \
                         f'Text = {data[word]["wiki_full"]}'
                _temp_word_data.append(get_chatgpt(prompt, role))

                # chain of thought + beide
                prompt = f'Was ist die Definition von {word}? Nutze den folgenden Text und ' \
                         f'die folgenden Belege als Hilfe und erklär deine Gedankenschritte. ' \
                         f'Text = {data[word]["wiki_full"]}, Belege = {data[word]["dwds_con"]}'
                _temp_word_data.append(get_chatgpt(prompt, role))
            else:
                _temp_word_data.append('Kein Wiki-Eintrag')
                _temp_word_data.append('Kein Wiki-Eintrag')

            # RAG
                # dwds Belege
            role = f'Lies die folgenden Belege und definiere auf Basis der Belege danach das Wort {word}.'
            prompt = f'Text = Belege = {data[word]["dwds_con"]}'
            _temp_word_data.append(get_chatgpt(prompt, role))

            if data[word]['wiki_full'] != 'no entry':
                # wikipedia article
                role = f'Lies den folgenden Text und definiere auf Basis des Textes das Wort {word}.'
                prompt = f'Text = {data[word]["wiki_full"]}'
                _temp_word_data.append(get_chatgpt(prompt, role))

                # beides
                role = f'Lies die folgenden Text und die Belege und definiere auf Basis des Textes und ' \
                       f'der Belege danach das Wort {word}.'
                prompt = f'Text = {data[word]["wiki_full"]}, Belege = {data[word]["dwds_con"]}'
                _temp_word_data.append(get_chatgpt(prompt, role))
            else:
                _temp_word_data.append('Kein Wiki-Eintrag')
                _temp_word_data.append('Kein Wiki-Eintrag')

            _temp_word_data_df = pandas.DataFrame([_temp_word_data],
                                                  columns=['Word',
                                                           'Wiki_def',
                                                           'DWDS_def',
                                                           'Zero',
                                                           'Zero_dwds',
                                                           'Zero_wiki',
                                                           'Zero_both',
                                                           'Few',
                                                           'Few_dwds',
                                                           'Few_wiki',
                                                           'Few_both',
                                                           'CoT',
                                                           'CoT_dwds',
                                                           'CoT_wiki',
                                                           'CoT_both',
                                                           'RAG_dwds',
                                                           'RAG_wiki',
                                                           'RAG_both'])
            all_data_df = pandas.concat([all_data_df, _temp_word_data_df], ignore_index=True)

        bar()

        all_data_df.to_csv('data.csv', sep='|', mode='a', encoding='utf-8')
        print('Data can be found in data.csv')


def secondary_prompting(search_words, examples, roles, data):
    """Retrieves Chat GPT reaction to different prompting options."""

    pass


if __name__ == '__main__':
    search_words = ['Ruhender Ball', 'Tikitaka', 'Chancentod', 'Mausohr', 'Gefrett']
    examples = {'Trauma': 'Ereignis, durch das ein Organismus (durch mechanische Gewalteinwirkung, '                    # Wort des Tages 13.07.
                          'Verätzung, Vergiftung, Verbrennung o.Ä.) geschädigt oder verletzt '
                          'wird; die Schädigung oder Verletzung selbst; schwere psychische Erschütterung',
                'Vlies': 'Fell des Schafes und die nach der Schur in ihrer natürlichen Form zusammenhängende Wolle',    # Wort des Tages 11.07.
                'Filmproduzent': 'Person, Firma oder Einrichtung, die die wirtschaftliche und technische '
                                 'Verantwortung für die Produktion eines Films trägt, die Dreharbeiten usw. '
                                 'organisiert und inhaltlich Einfluss nimmt'}                                           # Wort des Tages 07.07.
    roles = ['Du bist ein hilfreicher Assistent.',
             'Du bist ein linguist und arbeitest an einem lexikalischen Wörterbuch. '
             'Deine Aufgabe ist es, Wörter zu definieren. Die Definitionen dürfen nicht länger als 30 Wörter sein.']
    data = get_word_content(search_words)

    main_prompting(search_words, examples, roles, data)
