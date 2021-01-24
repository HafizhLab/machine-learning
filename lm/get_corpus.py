import requests
import json

response = requests.get("http://api.alquran.cloud/v1/quran/quran-simple")
data = json.loads(response.text)
surahs = data['data']['surahs']

with open('alquran_korpus.txt', 'w+') as the_file:
    for surah in surahs:
        for ayah in surah['ayahs']:
            the_file.write(ayah['text'] + '\n')
