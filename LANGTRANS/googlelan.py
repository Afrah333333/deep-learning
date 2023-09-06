from googletrans import Translator

import googletrans

translator = Translator()
result = translator.translate('come' , src='en',dest='ja')
print(result)
print(googletrans.LANGUAGES)