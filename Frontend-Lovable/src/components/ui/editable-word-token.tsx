import React, { useEffect, useMemo, useRef, useState, forwardRef, useImperativeHandle } from "react";

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
  toggle_on: boolean;
};

export interface EditableWordTokenProps {
  token: WordToken;
  /** Called whenever the text changes. Receives the updated token. */
  onChange?: (next: WordToken) => void;
  /** Called when the token loses focus (onBlur). Receives the current token. */
  onBlur?: (token: WordToken) => void;
  /** Called when the token gains focus. */
  onFocus?: () => void;
  className?: string;
  /** If true, disallow spaces/newlines inside the token. Default: true */
  singleWord?: boolean;
}

export interface EditableWordTokenRef {
  focus: () => void;
}

/**
 * Simple, editable token component.
 * - Renders the token's `text` inline.
 * - Holds an initial `originalText` (from `token.original_text`).
 * - Holds an initial `predicted` (from `token.predicted`).
 * - When the current text differs from the original, toke.isEdited is set to true.
 * - The token is rendered in GREEN if toke.predicted is true, BLUE if toke.isEdited is true, otherwise in black.
 */
const EditableWordToken = forwardRef<EditableWordTokenRef, EditableWordTokenProps>(({
  token,
  onChange,
  onBlur,
  onFocus,
  className,
  singleWord = true,
}, forwardedRef) => {


  const emitChange = () => {
    const hasChanged = token.text !== token.original_text;
    const next: WordToken = {
      ...token,
      text: token.text,
      isEdited: hasChanged,
      to_synth: token.predicted || hasChanged || token.toggle_on,
      
    };
    onChange?.(next);
  };

  const internalRef = useRef<HTMLSpanElement>(null);

  useImperativeHandle(forwardedRef, () => ({
    focus: () => {
      internalRef.current?.focus();
    },
  }));

  // Sanitize and emit on each input
  const handleInput = () => {
    const el = internalRef.current;
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
        if (internalRef.current && internalRef.current.innerText !== next) {
          internalRef.current.innerText = next;
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

  const handleBlur: React.FocusEventHandler<HTMLSpanElement> = () => {
    onBlur?.(token);
  };

  const handleFocus: React.FocusEventHandler<HTMLSpanElement> = () => {
    onFocus?.();
  };

  // Ensure DOM reflects the current controlled text
  useEffect(() => {
    if (internalRef.current && (internalRef.current.innerText ?? "") !== token.text) {
      internalRef.current.innerText = token.text;
    }
  }, [token.text]);

  return (
    <span
      ref={internalRef}
      role="textbox"
      contentEditable
      suppressContentEditableWarning
      onInput={handleInput}
      onKeyDown={handleKeyDown}
      onBlur={handleBlur}
      onFocus={handleFocus}
      className={[
        "inline-block rounded-md px-0.5",
        token.isEdited ? "text-blue-600 bg-blue-50" : token.predicted ? "text-green-600 bg-green-50" : token.toggle_on ? "text-red-600 bg-red-50" : "text-black",
        token.text === "" ? "min-w-[2ch]" : "",
        className || "",
      ].join(" ")}
      style={{
        // optional subtle transition for the highlight
        transition: "background-color 120ms ease, color 120ms ease",
      }}
      aria-label="Editable token"
    />
  );
});

EditableWordToken.displayName = "EditableWordToken";

export default EditableWordToken;
