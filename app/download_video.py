import yt_dlp

def download_video(video_url, output_path='data/video.mp4'):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    download_video(video_url)
    print("✅ Video downloaded to data/video.mp4")
