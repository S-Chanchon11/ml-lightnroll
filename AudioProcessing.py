# split audio file into chuck for every frequency change??
# split audio file into chuck for every time signature
# how to know specific time? user input time signature and BPM
# 


import librosa
from pydub import AudioSegment

def cut_audio(input_file, output_prefix, cut_points, cut_duration_seconds):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Convert cut duration to milliseconds
    cut_duration_ms = cut_duration_seconds * 1000

    # Iterate over the cut points and create audio chunks
    for i, cut_point in enumerate(cut_points):
        start_time = int(cut_point * 1000)  # Convert seconds to milliseconds
        end_time = start_time + cut_duration_ms

        # Ensure the end time does not exceed the audio duration
        end_time = min(end_time, len(audio))

        # Extract the corresponding portion of the audio
        chunk = audio[start_time:end_time]

        # Save the chunk as a new audio file
        output_file = f"guitar_chord/Test/temp/{output_prefix}_{i}.wav"
        chunk.export(output_file, format="wav")
        print(f"Saved {output_file}")

input_audio_file = "guitar_chord/All/Multi/G_E_1.wav"
output_file_prefix = "out"
cut_points = [0.13931973, 0.67337868, 1.76471655, 2.73995465, 3.76163265,
       4.82975057, 5.87464853, 6.94276644, 7.89478458]
cut_duration_seconds = 1

cut_audio(input_audio_file, output_file_prefix, cut_points, cut_duration_seconds)