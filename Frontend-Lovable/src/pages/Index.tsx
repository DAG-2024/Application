import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { AudioPlayer } from "@/components/AudioPlayer";
import { RotateCcw, Loader2, Download } from "lucide-react";
import type { WordToken } from "@/types";
import TokenEditor from "@/components/ui/tokens-editor";

const Index = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [originalAudioUrl, setOriginalAudioUrl] = useState<string | null>(null);
  const [transcriptionResults, setTranscriptionResults] = useState<WordToken[]>([]);
  const [isGeneratingTranscription, setIsGeneratingTranscription] = useState(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  const [finalAudioUrl, setFinalAudioUrl] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    return () => {
      if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
    };
  }, [originalAudioUrl]);


  const onChooseFile = () => {
    fileInputRef.current?.click();
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
    const url = URL.createObjectURL(file);
    setAudioFile(file);
    setOriginalAudioUrl(url);
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
    console.log(transcriptionResults);

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

      const response = await fetch("http://localhost:9001/fix-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Extract the fixed audio URL from the response
      const fixedUrl = data.fixed_url;
      const segments = data.segments;
      console.log(segments);
      // The backend returns file:// URLs, but we need to serve them properly
      // Convert file:// URLs to a proper endpoint that serves the audio files
      let processedUrl = fixedUrl;
      if (fixedUrl.startsWith('file://')) {
        const filePath = fixedUrl.replace('file://', '');
        const fileName = filePath.split('/').pop();
        processedUrl = `http://localhost:9001/audio/${fileName}`;
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
        <section aria-labelledby="tokens-heading">
          <h2 id="tokens-heading" className="sr-only">Token editor</h2>
          <div className="rounded-lg border bg-card p-4 shadow-elegant">
            {transcriptionResults.length > 0 ? (
              <TokenEditor
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
         {/* Original audio section */}
        <section aria-labelledby="player-heading" className="space-y-3">
          <div className="rounded-lg border bg-card p-4 shadow-elegant">
          <h2 id="player-heading" className="text-lg font-semibold mb-3">Original Audio</h2>
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <AudioPlayer src={originalAudioUrl} />
            </div>
            {originalAudioUrl ? (
              <Button asChild variant="white" size="icon" className="transition-smooth">
                <a href={originalAudioUrl} download={audioFile?.name || "audio-file"}>
                  <Download className="h-4 w-4" />
                </a>
              </Button>
            ) : (
              <Button variant="white" size="icon" disabled>
                <Download className="h-4 w-4" />
              </Button>
            )}
          </div>
          </div>
        </section>

        {/* Final processed audio section */}
        {(finalAudioUrl || isProcessingAudio) && (
          <section aria-labelledby="final-player-heading" className="space-y-3">
            <div className="rounded-lg border bg-card p-4 shadow-elegant">
            <h2 id="final-player-heading" className="text-lg font-semibold mb-3">Processed Audio</h2>
            <div className="flex items-center gap-3">
              <div className="flex-1">
                {isProcessingAudio ? (
                  <div className="flex items-center justify-center h-16 bg-muted rounded-lg">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : (
                  <AudioPlayer src={finalAudioUrl} />
                )}
              </div>
              {finalAudioUrl && !isProcessingAudio ? (
                <Button asChild variant="white" size="icon" className="transition-smooth">
                  <a href={finalAudioUrl} download="processed-audio.wav">
                    <Download className="h-4 w-4" />
                  </a>
                </Button>
              ) : (
                <Button variant="white" size="icon" disabled>
                  <Download className="h-4 w-4" />
                </Button>
              )}
            </div>
            </div>
          </section>
        )}

        
      </main>
    </div>
  );
};

export default Index;
