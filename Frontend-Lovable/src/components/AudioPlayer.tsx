import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Play, Pause } from "lucide-react";

interface AudioPlayerProps {
  src: string | null;
}

const formatTime = (s: number) => {
  if (!isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${sec}`;
};

export function AudioPlayer({ src }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressRef = useRef<HTMLDivElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTime = () => {
      setCurrentTime(audio.currentTime);
      if (audio.duration) setProgress((audio.currentTime / audio.duration) * 100);
    };
    const onLoaded = () => {
      setDuration(audio.duration || 0);
    };
    const onEnded = () => setIsPlaying(false);

    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("loadedmetadata", onLoaded);
    audio.addEventListener("ended", onEnded);
    return () => {
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("loadedmetadata", onLoaded);
      audio.removeEventListener("ended", onEnded);
    };
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    setIsPlaying(false);
    setProgress(0);
    setCurrentTime(0);
    setDuration(0);
    if (src) {
      // Ensure the new source loads
      audio.load();
    }
  }, [src]);

  const canControl = useMemo(() => Boolean(src), [src]);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio || !src) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      try {
        await audio.play();
        setIsPlaying(true);
      } catch (e) {
        // ignore autoplay errors
      }
    }
  };

  const seekTo = (clientX: number) => {
    const audio = audioRef.current;
    const progressBar = progressRef.current;
    if (!audio || !progressBar || !duration) return;

    const rect = progressBar.getBoundingClientRect();
    const clickX = clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, clickX / rect.width));
    const newTime = percentage * duration;
    
    audio.currentTime = newTime;
    setCurrentTime(newTime);
    setProgress(percentage * 100);
  };

  const handleProgressClick = (e: React.MouseEvent) => {
    if (!canControl) return;
    seekTo(e.clientX);
  };

  const handleProgressMouseDown = (e: React.MouseEvent) => {
    if (!canControl) return;
    setIsDragging(true);
    seekTo(e.clientX);
  };

  const handleProgressMouseMove = (e: MouseEvent) => {
    if (!isDragging || !canControl) return;
    seekTo(e.clientX);
  };

  const handleProgressMouseUp = () => {
    setIsDragging(false);
  };

  // Add global mouse event listeners for dragging
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleProgressMouseMove);
      document.addEventListener('mouseup', handleProgressMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleProgressMouseMove);
        document.removeEventListener('mouseup', handleProgressMouseUp);
      };
    }
  }, [isDragging, canControl, duration]);

  return (
    <div className="w-full rounded-lg border bg-card p-4 shadow-elegant">
      <div className="flex items-center gap-4">
        <Button variant="white" onClick={togglePlay} disabled={!canControl} className="transition-smooth">
          {isPlaying ? (
            <>
              <Pause />
              Pause
            </>
          ) : (
            <>
              <Play />
              Play
            </>
          )}
        </Button>
        <div className="flex-1">
          <div 
            ref={progressRef}
            className={`relative h-2 bg-secondary rounded-full cursor-pointer transition-colors ${
              canControl ? 'hover:bg-secondary/80' : 'cursor-not-allowed'
            }`}
            onClick={handleProgressClick}
            onMouseDown={handleProgressMouseDown}
          >
            <div 
              className="absolute top-0 left-0 h-full bg-primary rounded-full transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
            {/* Circular thumb at the end of progress */}
            <div 
              className="absolute top-1/2 w-4 h-4 bg-primary rounded-full border-2 border-background shadow-md transform -translate-y-1/2 transition-all duration-100"
              style={{ left: `calc(${progress}% - 8px)` }}
            />
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>
      </div>
      {/* Hidden audio element */}
      <audio ref={audioRef} preload="metadata">
        {src && <source src={src} />}
      </audio>
    </div>
  );
}
