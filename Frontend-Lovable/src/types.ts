// types.ts
export type WordToken = {
  start: number;
  end: number;
  text: string;
  original_text: string;
  isEdited: boolean;
  predicted: boolean;
  to_synth: boolean;
  is_speech: boolean;
  synth_path?: string | null; // may be absent or null
};
