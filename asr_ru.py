import os
import torch
from huggingface_hub import login
import nemo.collections.asr as nemo_asr
from datetime import datetime

# Установка токена Hugging Face и авторизация
HUGGINGFACE_TOKEN = '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'
login(token=HUGGINGFACE_TOKEN)

# Настройка PyTorch для использования GPU, если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)

# Убедимся, что директория 'answers_audio' существует, если нет, создадим её
os.makedirs('answers_audio', exist_ok=True)

# Загрузка модели ASR (Automatic Speech Recognition)
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")

# Транскрибирование аудио файла
audio_file = 'audios/я_разработчик.wav'
transcriptions = asr_model.transcribe([audio_file])
first_element_string = " ".join(transcriptions[0])
print(f"Транскрибированный текст: {first_element_string}")

# Загрузка модели TTS (Text-to-Speech)
local_file = 'model.pt'
model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

# Функция для сохранения аудио с уникальным именем файла
def save_audio(text, response):
    sample_rate = 48000
    speaker = 'baya'
    if text in first_element_string:
        # Генерация аудио для ответа
        audio_path = model.save_wav(text=response, speaker=speaker, sample_rate=sample_rate)

        # Создание уникального имени файла на основе текущей даты и времени
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}.wav"
        file_path = os.path.join('answers_audio', filename)
        
        # Перемещение сгенерированного файла в директорию 'answers_audio' с уникальным именем
        os.rename(audio_path, file_path)
        print(f"Аудио сохранено в: {file_path}")

# Проверка условий и сохранение файлов
save_audio("привет я разработчик", "сегодня у меня выходной")
save_audio("я сегодня не приду домой", "Ну и катись отсюда")
