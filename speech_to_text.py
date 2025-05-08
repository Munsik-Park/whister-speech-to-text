import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import scipy.io.wavfile as wav
import librosa

def record_audio(duration=5, output_file="temp_recording.wav"):
    print(f"* 녹음 시작 ({duration}초)")
    
    # 녹음 설정
    sample_rate = 16000
    recording = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32')
    
    # 녹음 대기
    sd.wait()
    
    print("* 녹음 완료")
    
    # 파일 저장
    wav.write(output_file, sample_rate, recording)
    
    return output_file

def main():
    # 모델과 프로세서 로드
    print("모델 로딩 중...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    print("모델 로딩 완료!")
    
    while True:
        input("Enter를 눌러 녹음을 시작하세요...")
        audio_file = record_audio(duration=5)
        
        # 오디오 파일 로드 및 변환
        print("음성 인식 중...")
        audio, sampling_rate = librosa.load(audio_file, sr=16000)
        input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
        
        # 한국어로 강제 설정
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ko",
            task="transcribe",
            no_timestamps=True
        )
        
        # 텍스트 생성
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448,
            num_beams=5,
            temperature=0.0
        )
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        print("인식된 텍스트:", transcription[0])
        
        if input("계속하시겠습니까? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()