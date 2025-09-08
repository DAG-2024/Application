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
  toggle_on: boolean;
};

export type Segment = {
  start: number;
  end: number;
  source: "orig" | "synth";
  tokens: number[];
  text: string;
  src_start?: number | null;
  src_end?: number | null;
  overlap_in: number;
  overlap_out: number;
};
