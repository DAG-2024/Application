import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { AudioPlayer } from "@/components/AudioPlayer";

const Index = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const { toast } = useToast();

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const onChooseFile = () => fileInputRef.current?.click();

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const url = URL.createObjectURL(file);
    setAudioFile(file);
    setAudioUrl(url);
  };

  const onSubmit = () => {
    toast({
      title: "Submitted",
      description: "Your notes have been submitted.",
    });
  };

  return (
    <div>
      <header className="w-full bg-subtle-gradient">
        <div className="container mx-auto max-w-3xl px-4 py-10">
          <h1 className="text-3xl font-bold tracking-tight mb-6">
            DAG - Audio Restorer
          </h1>
          <Button variant="white" onClick={onChooseFile} className="transition-smooth">
            Upload audio file
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={onFileChange}
            className="hidden"
            aria-label="Upload audio file"
          />
        </div>
      </header>

      <main className="container mx-auto max-w-3xl px-4 pb-16 space-y-6">
        <section aria-labelledby="notes-heading">
          <h2 id="notes-heading" className="sr-only">Notes editor</h2>
          <div className="rounded-lg border bg-card p-4 shadow-elegant">
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Write your notes here..."
              className="min-h-[180px] bg-card"
            />
          </div>
        </section>

        <section>
          <Button onClick={onSubmit} variant="default" className="transition-smooth">
            Submit
          </Button>
        </section>

        <section aria-labelledby="player-heading" className="space-y-3">
          <h2 id="player-heading" className="sr-only">Audio player</h2>
          <AudioPlayer src={audioUrl} />
        </section>

        <section>
          {audioUrl ? (
            <Button asChild variant="white" className="transition-smooth">
              <a href={audioUrl} download={audioFile?.name || "audio-file"}>
                Download audio
              </a>
            </Button>
          ) : (
            <Button variant="white" disabled>
              Download audio
            </Button>
          )}
        </section>
      </main>
    </div>
  );
};

export default Index;
