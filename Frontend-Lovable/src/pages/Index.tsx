import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { AudioPlayer } from "@/components/AudioPlayer";
import { RotateCcw, Loader2 } from "lucide-react";
import type { WordToken } from "@/types";
import TokenRow from "@/components/ui/tokens-editor"; // <-- adjust path as needed

const Index = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcriptionResults, setTranscriptionResults] = useState<WordToken[]>([]);
  const [isGeneratingTranscription, setIsGeneratingTranscription] = useState(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  const [finalAudioUrl, setFinalAudioUrl] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const onChooseFile = () => {
    fileInputRef.current?.click();
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const url = URL.createObjectURL(file);
    setAudioFile(file);
    setAudioUrl(url);
  };

  const onGenerateTranscription = async () => {
    if (!audioFile) {
      toast({
        title: "No audio file",
        description: "Please upload an audio file first.",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingTranscription(true);
    toast({
      title: "Generating transcription",
      description: "Transcription is being generated...",
    });

    try {
      const formData = new FormData();
      formData.append("file", audioFile);

      const response = await fetch("http://127.0.0.1:9002/feed-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Backend returns: { wordtokens: [...] }
      const raw: WordToken[] = data.wordtokens || [];

      // Ensure original_text exists; default it to the initial text
      const normalized: WordToken[] = raw.map((t) => ({
        ...t,
        original_text: t.original_text ?? t.text,
        isEdited: false,
        predicted: t.to_synth,
      }));

      setTranscriptionResults(normalized);

      toast({
        title: "Transcription complete",
        description: `Loaded ${normalized.length} tokens.`,
      });
    } catch (error) {
      console.error("Error generating transcription:", error);
      toast({
        title: "Error",
        description: "Failed to generate transcription. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingTranscription(false);
    }
  };

  const onSubmit = async () => {
    if (!audioFile || transcriptionResults.length === 0) {
      toast({
        title: "Missing data",
        description: "Please upload an audio file and generate transcription first.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessingAudio(true);
    toast({
      title: "Processing audio",
      description: "Fixing audio with transcription edits...",
    });

    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      
      // Convert transcription results to JSON string
      const payload = JSON.stringify(transcriptionResults.map(token => ({
        start: token.start,
        end: token.end,
        text: token.text,
        to_synth: token.to_synth,
        is_speech: token.is_speech || true,
        synth_path: token.synth_path || null
      })));
      
      formData.append("payload", payload);

      const response = await fetch("http://0.0.0.0:9001/fix-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Extract the fixed audio URL from the response
      const fixedUrl = data.fixed_url;
      
      // The backend returns file:// URLs, but we need to serve them properly
      // Convert file:// URLs to a proper endpoint that serves the audio files
      let processedUrl = fixedUrl;
      if (fixedUrl.startsWith('file://')) {
        const filePath = fixedUrl.replace('file://', '');
        const fileName = filePath.split('/').pop();
        processedUrl = `http://0.0.0.0:9001/audio/${fileName}`;
      }
      
      setFinalAudioUrl(processedUrl);

      toast({
        title: "Audio processing complete",
        description: "Your audio has been processed successfully!",
      });
    } catch (error) {
      console.error("Error processing audio:", error);
      toast({
        title: "Error",
        description: "Failed to process audio. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessingAudio(false);
    }
  };

  return (
    <div>
      <header className="w-full bg-subtle-gradient">
        <div className="container mx-auto max-w-3xl px-4 py-10">
          <h1 className="text-3xl font-bold tracking-tight mb-6">
            DAG - Audio Restorer
          </h1>
          <div className="flex gap-3">
            <Button variant="white" onClick={onChooseFile} className="transition-smooth">
              Upload audio file
            </Button>
            <Button
              variant="outline"
              onClick={onGenerateTranscription}
              disabled={!audioFile || isGeneratingTranscription}
              className="transition-smooth flex items-center gap-2"
            >
              {isGeneratingTranscription ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              {isGeneratingTranscription ? "Generating..." : "Generate transcription"}
            </Button>
          </div>
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
        {/* --- Token editor (replaces the textarea) --- */}
        <section aria-labelledby="tokens-heading">
          <h2 id="tokens-heading" className="sr-only">Token editor</h2>
          <div className="rounded-lg border bg-card p-4 shadow-elegant">
            {transcriptionResults.length > 0 ? (
              <TokenRow
                tokens={transcriptionResults}
                onChange={setTranscriptionResults}
              />
              
            ) : (
              <p className="text-sm text-muted-foreground">
                Upload an audio file and generate a transcription that you can edit.
              </p>
            )}
          </div>
        </section>

        <section>
          <Button 
            onClick={onSubmit} 
            variant="default" 
            disabled={!audioFile || transcriptionResults.length === 0 || isProcessingAudio}
            className="transition-smooth flex items-center gap-2"
          >
            {isProcessingAudio ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : null}
            {isProcessingAudio ? "Processing..." : "Submit"}
          </Button>
        </section>

        <section aria-labelledby="player-heading" className="space-y-3">
          <h2 id="player-heading" className="sr-only">Audio player</h2>
          <AudioPlayer src={audioUrl} />
        </section>

        {/* Final processed audio section */}
        {finalAudioUrl && (
          <section aria-labelledby="final-player-heading" className="space-y-3">
            <h2 id="final-player-heading" className="text-lg font-semibold">Processed Audio</h2>
            <div className="rounded-lg border bg-card p-4 shadow-elegant">
              <AudioPlayer src={finalAudioUrl} />
              <div className="mt-3">
                <Button asChild variant="white" className="transition-smooth">
                  <a href={finalAudioUrl} download="processed-audio.wav">
                    Download processed audio
                  </a>
                </Button>
              </div>
            </div>
          </section>
        )}

        <section>
          {audioUrl ? (
            <Button asChild variant="white" className="transition-smooth">
              <a href={finalAudioUrl} download={audioFile?.name || "audio-file"}>
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
