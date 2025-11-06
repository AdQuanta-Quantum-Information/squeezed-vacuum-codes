# `matplotlib` and `moviepy`` stuff
from moviepy import ImageClip, concatenate_videoclips
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from pathlib import Path

from typing import Optional, Generator, Iterable, TypeVar, Callable, overload, Type, Literal, TypeAlias
_T = TypeVar('_T')
_PathOrStrType = TypeVar('_PathOrStrType', Path, str)

if __name__ == "__main__":
    import sys
    sys.path.append("../../..")
    from src.utils import assertions, arguments, strings, files, prints
    from src import code_paths

else:
    from .. import assertions, arguments, strings, files, prints
    from ... import code_paths

from .matplotlib_support import clean_figure_memory

DEFAULT_VIDEOS_FOLDER = code_paths.outputs / "videos"
SupportedVideoCodecsType : TypeAlias = Literal["mp4", "avi"]


def _parse_path(path: Path|str, target: Type[_PathOrStrType] = str) -> _PathOrStrType:
    if isinstance(path, Path):
        if target is str:
            return path.absolute().__str__()  # type: ignore
        elif target is Path:
            return path  # type: ignore
        
    elif isinstance(path, str):
        if target is str:
            return path  # type: ignore
        return Path(path)  # type: ignore
    
    return path  # type: ignore


class VideoRecorder():
    def __init__(self, fps:float=10.0, folder_full_path:Path|str=DEFAULT_VIDEOS_FOLDER, codec:SupportedVideoCodecsType = "mp4") -> None:
        if ImageClip is None:
            raise ImportError("")
        self.fps : float = fps
        self.folder_full_path : Path = _parse_path(folder_full_path, target=Path)  #type: ignore
        self.frames_dir : Path = self._reset_temp_folders_dir()
        self.frames_duration : list[int] = []
        self.frames_counter : int = 0
        self.codec : SupportedVideoCodecsType = codec

    def capture(self, fig:Optional[Figure]=None, duration_in_frames:int|None=None, duration_in_seconds:float|None=None, clear_figure_after:bool=False, dpi:int|None=None)->None:
        # Complete missing inputs:
        if fig is None:
            fig = plt.gcf()
        duration_in_frames = _derive_duration_in_frames(duration_in_frames, duration_in_seconds, self.fps)
        # Check inputs:
        assertions.integer(duration_in_frames, reason=f"duration must be an integer - meaning the number of frames to repeat a single shot")
        # Prepare data
        fullpath = self.crnt_frame_path
        # save file:
        fig.savefig(fullpath, dpi=dpi)
        # Update:
        self.frames_counter += 1
        self.frames_duration.append(duration_in_frames)

        ## Clear figure memory if requested:
        if clear_figure_after:
            clean_figure_memory(fig)

    def extend_prev_frame(self, duration_in_frames:int|None=None, duration_in_seconds:float|None=None)->None:
        return self.extend_frame_at_index(-2, duration_in_frames, duration_in_seconds)
    
    def extend_frame_at_index(self, index:int, duration_in_frames:int|None=None, duratuin_in_seconds:float|None=None)->None:
        duration_in_frames = _derive_duration_in_frames(duration_in_frames, duratuin_in_seconds, self.fps)
        self.frames_duration[index] += duration_in_frames

    def write_video(self, name:Optional[str]=None)->None:
        # Complete missing inputs:
        if name is None:
            name = strings.time_stamp()
        # Prepare folder for video:
        folder = self.folder_full_path
        files.force_folder_exists(folder)
        # Put clips together:
        clips_gen = self.image_clips()
        video_slides = concatenate_videoclips( list(clips_gen), method='chain' )
        
        # Write video file:
        fullpath = str(folder/name)+"."+self.codec
        video_slides.write_videofile(filename=fullpath, fps=self.fps)

    @property
    def crnt_frame_path(self) -> str:         
        crnt_frame_fullpath_str = self._get_frame_path(self.frames_counter).__str__()
        return  crnt_frame_fullpath_str+".png"

    def image_clips(self) -> Generator[ImageClip, None, None] :
        base_duration = 1/self.fps
        for img_path, frame_duration in zip( self.image_paths(), self.frames_duration, strict=True):
            image_fullpath_str = str(img_path)+".png"
            yield ImageClip(image_fullpath_str, duration=base_duration*frame_duration)

    def image_paths(self) -> Generator[Path, None, None] :
        for i in range(self.frames_counter):
            yield self._get_frame_path(i)

    def _get_frame_path(self, index:int) -> Path:
        frame_filename = "frame"+f"{index}"
        return self.frames_dir / frame_filename
    
    def _reset_temp_folders_dir(self) -> Path:
        videos_folder = self.folder_full_path
        temp_folder = "temp_frames__"+strings.time_stamp()
        frames_dir = videos_folder / temp_folder
        files.force_folder_exists(frames_dir)
        return frames_dir


def _derive_duration_in_frames(duration_in_frames:int|None, duration_in_time:float|None, fps:float) -> int:
    if duration_in_frames is not None and duration_in_time is not None:
        raise ValueError("Only one of `duration` and `time` should be provided")
    if duration_in_frames is not None:
        return duration_in_frames
    if duration_in_time is not None:
        return max(1, int(duration_in_time*fps))
    if duration_in_frames is None and duration_in_time is None:
        return 1


## ============================== ##
##      Independent functions:    ##
## ============================== ##


@overload
def record_video(iterable:Iterable[Figure], function:None, fps:float=30.0,  folder_full_path:Path|str=DEFAULT_VIDEOS_FOLDER, progress_bar:bool=True)->None: ...
@overload
def record_video(iterable:Iterable[_T], function:Callable[[_T], Figure], fps:float=30.0,  folder_full_path:Path|str=DEFAULT_VIDEOS_FOLDER, progress_bar:bool=True)->None: ...
def record_video(
    iterable:Iterable[_T],
    function:Callable[[_T], Figure]|None,
    fps:float=30.0, 
    folder_full_path:Path|str=DEFAULT_VIDEOS_FOLDER,
    progress_bar:bool=True
)->None:
    if function is None:
        _record_video_from_figures(iterable, fps=fps, folder_full_path=folder_full_path, progress_bar=progress_bar)
    else:
        _record_video_from_function(iterable, function, fps=fps, folder_full_path=folder_full_path, progress_bar=progress_bar)


def _try_to_get_length(iterable:Iterable)->int|None:
    try:
        return len(iterable)  #type: ignore
    except Exception:
        return None

def _record_video_from_function(
    iterable:Iterable[_T],
    function:Callable[[_T], Figure],
    fps:float=30.0, 
    folder_full_path:Path|str=DEFAULT_VIDEOS_FOLDER,
    progress_bar:bool=True
)->None:
    
    n = _try_to_get_length(iterable)
    figures = (function(item) for item in iterable)
    _record_video_from_figures(figures, fps=fps, folder_full_path=folder_full_path, progress_bar=progress_bar, length=n)


def _record_video_from_figures(
    figures:Iterable[Figure],
    fps:float=30.0, 
    folder_full_path:Path|str=DEFAULT_VIDEOS_FOLDER,
    progress_bar:bool=True,
    length:int|None=None
)->None:
    # Prepare recorder:
    recorder = VideoRecorder(fps=fps, folder_full_path=folder_full_path)
    ## Progress bar if applicable:
    # Check if a progress bar can be used:
    if length is None:
        length = _try_to_get_length(figures)
    # Get iterator:
    if length is not None and progress_bar:
        figures_iter = prints.ProgressBar(figures, prefix="Recording Video: ", suffix=" frames", _expected_length=length)
    else:
        figures_iter = figures
    
    ## Iterate and record:
    for fig in figures_iter:
        recorder.capture(fig)
    recorder.write_video()


def record_video_from_files(folders:list[Path|str], fps:float=30.0, result_fullpath:str|Path|None=None)->None:
    base_duration = 1/fps

    ## Parse inputs:
    if result_fullpath is None:
        result_fullpath = DEFAULT_VIDEOS_FOLDER/strings.time_stamp()
    result_fullpath = _parse_path(result_fullpath, target=str)  
    # If filename does not have an extension, add `.mp4`:
    if not files.has_extension(result_fullpath):
        result_fullpath += ".mp4"


    ## Go through folders:
    image_clips : list[ImageClip] = []
    for folder in folders:
        folder = _parse_path(folder, target=str)
        for file in files.get_all_files_fullpath_in_folder(folder):        
            clip = ImageClip(file, duration=base_duration)
            image_clips.append(clip)

    video_slides = concatenate_videoclips( image_clips, method='chain')

    # Write video file:
    video_slides.write_videofile(result_fullpath, fps=fps)





def as_main(
    folders=["temp_frames__2024.11.04_13.48.45", "temp_frames__2024.11.04_15.58.16"]
):
    fullpaths = [DEFAULT_VIDEOS_FOLDER / folder for folder in folders]
    record_video_from_files(fullpaths, fps=30.0)
    print("Done")
    


if __name__ == "__main__":
    as_main()