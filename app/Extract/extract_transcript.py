from faster_whisper import WhisperModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Paths import video_path,output_text_path


def transcribe_audio(video_path,output_txt_path=output_text_path ):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    segments, info = model.transcribe(video_path)

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for segment in segments:
            start = segment.start
            end = segment.end
            text = segment.text
            print(text)
            f.write(f"{start:.2f} --> {end:.2f}\n{text}\n\n")

    print(f"✅ Transcript saved to {output_txt_path}")


def transcribe_audio_multi_sentence(video_path, output_txt_path=output_text_path,number_of_sentences=5):
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Model output is initialized!")

    segments, info = model.transcribe(video_path)

    print(f"Model output is ready!")

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        segment_groups=[]
        curr_seg = []
        
        print("Grouping Segments started!")

        for segment in segments:
            if(len(curr_seg)<number_of_sentences):
                curr_seg.append(segment)
            else:
                segment_groups.append(curr_seg)
                curr_seg=[]
                print(f"Added group = {len(segment_groups)}")

        print("Grouping Segments ended!")

        if len(curr_seg)>0:
            segment_groups.append(curr_seg)

        print("Merging Groups started!")
        
        for group in segment_groups:
            start = group[0].start
            end = group[-1].end
            text = ""

            for segment in group:
                text += segment.text
            
            f.write(f"{start:.2f} --> {end:.2f}\n{text}\n\n")

    print(f"✅ Transcript saved to {output_txt_path}")


if __name__ == "__main__":
    #transcribe_audio(video_path)
    transcribe_audio_multi_sentence(video_path)
