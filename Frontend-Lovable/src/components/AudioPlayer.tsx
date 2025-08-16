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
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

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
          <Progress value={progress} />
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
