import moviepy.editor as mp
import numpy as np
import soundfile as sf
import librosa


class AudioModule:

    @classmethod
    def extract_audio_from_video(cls, video_path: str, saved_path: str) -> str:
        """
        extract audio from video
        :param video_path: path of video
        :param saved_path: where the audio must be saved
        :return:
        """
        mp.VideoFileClip(video_path).audio.write_audiofile(saved_path)
        return saved_path

    @classmethod
    def get_audio_data(cls, file_path: str, mono: bool = True, sampling_rate=None):
        """
        read audio from file and get the data of audio
        :param sampling_rate:
        :param mono:
        :param file_path:
        :return:
        """
        try:
            array, sr = sf.read(file_path)
            array = array.T

            if mono:
                array = librosa.to_mono(array)
            if sampling_rate and sampling_rate != sr:
                array = librosa.resample(array, orig_sr=sampling_rate, target_sr=sampling_rate)
                sr = sampling_rate

            return {"path": file_path, "array": array, "sampling_rate": sr}

        except Exception:
            return {"path": file_path, "array": np.ndarray([]),
                    "sampling_rate": 16000 if sampling_rate is None else sampling_rate}


