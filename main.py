import os
import requests
import json

from model.bow import BowModel

MODEL_ID = '5fc3925cfebe5bfa49164662'
API_KEY = 'jmWpkStPXiR3T6VJknhSLSow0FPNLkDdF1v76HnuOvI'
BASE_API_URL = 'https://ai-api-stag.cofacts.org/v1'
TEST = True

config = json.loads(open('./model-config.json', 'r').read())

DEFAULT_CATEGORY_MAPPING = dict([(int(k), v)
                                 for k, v in config['categoryMapping'].items()])


def predict():
    model = BowModel()
    while True:
        get_tasks_url = f'{BASE_API_URL}/tasks?modelId={MODEL_ID}&apiKey={API_KEY}'
        if TEST:
            get_tasks_url += '&test=1'

        tasks = requests.get(get_tasks_url).json()

        # print(tasks)

        if len(tasks) == 0:
            break

        result = []

        count = 0

        for task in tasks:
            text = task['content']
            count += 1

            # to avoid oversized requests
            if count > 100:
                break

            category = DEFAULT_CATEGORY_MAPPING[model.predict_text(text)[0]]

            temp = {
                'id': task['id'],
                'result': {
                    'prediction': {
                        'confidence': {}
                    },
                    'time': 1500000
                }
            }
            temp['result']['prediction']['confidence'][category] = 1.0
            result.append(temp)

        send_result = requests.post(f'{BASE_API_URL}/tasks', json=result)
        print(send_result.text)

        if TEST:
            break


def register():
    register_url = f'{BASE_API_URL}/models'

    result = requests.post(register_url, data={
        "name": "rumors-ai-bow",
        "realTime": False,
        "categoryMapping": DEFAULT_CATEGORY_MAPPING
    })

    print(result)


def main():
    if os.getenv('CFA_ACTION') == 'register':
        register()
    else:
        predict()


if __name__ == '__main__':
    main()
