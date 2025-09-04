import React, { useEffect, useMemo, useRef, useState } from "react";

export type WordToken = {
  start: number;
  end: number;
  text: string;
  original_text: string;
  isEdited: boolean;
  predicted: boolean;
  to_synth: boolean;
  is_speech: boolean;
  synth_path?: string | null;
};

export interface EditableWordTokenProps {
  token: WordToken;
  /** Called whenever the text changes. Receives the updated token. */
  onChange?: (next: WordToken) => void;
  className?: string;
  /** If true, disallow spaces/newlines inside the token. Default: true */
  singleWord?: boolean;
}

/**
 * Simple, editable token component.
 * - Renders the token's `text` inline.
 * - Holds an initial `originalText` (from `token.original_text`).
 * - Holds an initial `predicted` (from `token.predicted`).
 * - When the current text differs from the original, toke.isEdited is set to true.
 * - The token is rendered in GREEN if toke.predicted is true, BLUE if toke.isEdited is true, otherwise in black.
 */
export default function EditableWordToken({
  token,
  onChange,
  className,
  singleWord = false,
}: EditableWordTokenProps) {


  const emitChange = () => {
    const hasChanged = token.text !== token.original_text;
    const next: WordToken = {
      ...token,
      text: token.text,
      isEdited: hasChanged,
      to_synth: token.predicted || hasChanged,
      
    };
    onChange?.(next);
  };

  const ref = useRef<HTMLSpanElement>(null);

  // Sanitize and emit on each input
  const handleInput = () => {
    const el = ref.current;
    if (!el) return;
    let next = el.innerText ?? "";
    if (singleWord) {
      // Remove spaces & newlines for single-token editing
      next = next.replace(/\s+/g, "");
    } else {
      // Collapse newlines to spaces for nicer inline editing
      next = next.replace(/\n+/g, " ");
    }
    if (el.innerText !== next) {
      // Normalize the DOM if we stripped characters
      // (defer to keep caret sane)
      requestAnimationFrame(() => {
        if (ref.current && ref.current.innerText !== next) {
          ref.current.innerText = next;
        }
      });
    }
    if (next !== token.text) {
        token.text = next;
        emitChange();
    }
  };

  const handleKeyDown: React.KeyboardEventHandler<HTMLSpanElement> = (e) => {
    if (!singleWord) return;
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
    }
  };

  // Ensure DOM reflects the current controlled text
  useEffect(() => {
    if (ref.current && (ref.current.innerText ?? "") !== token.text) {
      ref.current.innerText = token.text;
    }
  }, [token.text]);

  return (
    <span
      ref={ref}
      role="textbox"
      contentEditable
      suppressContentEditableWarning
      onInput={handleInput}
      onKeyDown={handleKeyDown}
      className={[
        "inline-block rounded-md px-0.5",
        token.isEdited ? "text-blue-600 bg-blue-50" : token.predicted ? "text-green-600 bg-green-50" : "text-black",
        className || "",
      ].join(" ")}
      style={{
        // optional subtle transition for the highlight
        transition: "background-color 120ms ease, color 120ms ease",
      }}
      aria-label="Editable token"
    />
  );
}
