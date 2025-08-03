import ffmpeg
import OpenGL.GL as GL

from tools import working_dir, param, np

class MovieFFmpeg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width//2 * 2 #libx264（H.264エンコーダ）では偶数でないとエラー
        self.height = height//2 * 2
        self.total_frame = int(param.movie.fps * param.movie.total_sec)
        self.complete_flag = False

        movie_filepath = working_dir / param.save_dir / param.movie.filename
        movie_filepath.parent.mkdir(parents=True, exist_ok=True) #ディレクトリ作成
        self.ffmpeg_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                    s=f"{self.width}x{self.height}",
                    framerate=param.movie.fps)
            .output(str(movie_filepath), pix_fmt='yuv420p', vcodec='libx264')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True) # , pipe_stdout=True, pipe_stderr=True
        )
        
        print("動画保存開始：ウィンドウサイズを変更しないでください")

    def step(self, frame_count: int) -> None:
        if self.complete_flag:
            return
        if frame_count < self.total_frame:
            # CuPyの場合はasnumpyでCPUメモリへ変換
            data = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 4))
            image = np.flip(image, axis=0)[..., :3]  # 上下反転 & RGB
            image = np.ascontiguousarray(image, dtype=np.uint8)
            self.ffmpeg_proc.stdin.write(image.tobytes())
        elif frame_count >= self.total_frame:
            self.ffmpeg_proc.stdin.close()
            self.ffmpeg_proc.wait()
            print("動画保存完了！ ウィンドウサイズ変更OKです")
            self.complete_flag = True