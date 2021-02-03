import requests

from model.bow import BowModel

MODEL_ID = '5fc3925cfebe5bfa49164662'
API_KEY = 'jmWpkStPXiR3T6VJknhSLSow0FPNLkDdF1v76HnuOvI'
BASE_API_URL = 'https://ai-api-stag.cofacts.org/v1'
TEST = True

DEFAULT_CATEGORY_MAPPING = {
    0: 'kj287XEBrIRcahlYvQoS',  # 中國影響力
    1: 'kz3c7XEBrIRcahlYxAp6',  # 性少數與愛滋病
    2: 'lD3h7XEBrIRcahlYeQqS',  # 女權與性別刻板印象
    3: 'lT3h7XEBrIRcahlYugqq',  # 保健秘訣、食品安全
    4: 'lj2m7nEBrIRcahlY6Ao_',  # 基本人權問題
    5: 'lz2n7nEBrIRcahlYDgri',  # 農林漁牧政策
    6: 'mD2n7nEBrIRcahlYLAr7',  # 能源轉型
    7: 'mT2n7nEBrIRcahlYTArI',  # 環境生態保護
    8: 'mj2n7nEBrIRcahlYdArf',  # 優惠措施、新法規、政策宣導
    9: 'mz2n7nEBrIRcahlYnQpz',  # 科技、資安、隱私
    10: 'nD2n7nEBrIRcahlYwQoW',  # 免費訊息詐騙
    11: 'nT2n7nEBrIRcahlY6QqF',  # 有意義但不包含在以上標籤
    12: 'nj2n7nEBrIRcahlY-gpc',  # 無意義
    13: 'nz2o7nEBrIRcahlYBgqQ',  # 廣告
    14: 'oD2o7nEBrIRcahlYFgpm',  # 只有網址其他資訊不足
    15: 'oT2o7nEBrIRcahlYKQoM',  # 政治、政黨
    16: 'oj2o7nEBrIRcahlYRAox'  # 轉發協尋、捐款捐贈
}

def main():
  model = BowModel()
  while True:
    get_tasks_url = f'{BASE_API_URL}/tasks?modelId={MODEL_ID}&apiKey={API_KEY}'
    if TEST: get_tasks_url += '&test=1'

    tasks = requests.get(get_tasks_url).json()

    # print(tasks)

    if len(tasks) == 0: break

    result = []

    count = 0

    for task in tasks:
      text = task['content']
      count += 1

      # to avoid oversized requests
      if count > 100: break

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

    send_result = requests.post( f'{BASE_API_URL}/tasks', json=result)
    print(send_result.text)
    
    if TEST: break
  
  
if __name__ == '__main__':
  main()