import React from "react";
import type { WordToken } from "./editable-word-token";

export interface TokenSeparatorProps {
  /** Called when the separator is clicked. */
  onInsertToken: () => void;
  className?: string;
}

/**
 * A clickable vertical line that appears between tokens.
 * When clicked, it triggers the insertion of a new empty token at the corresponding position.
 */
export default function TokenSeparator({ onInsertToken, className }: TokenSeparatorProps) {
  const handleClick = () => {
    onInsertToken();
  };

  return (
    <div
      onClick={handleClick}
      className={[
        "w-1 h-6  hover:bg-blue-400 cursor-pointer transition-colors duration-150 rounded-sm",
        "flex items-center justify-center",
        "hover:shadow-sm",
        className || "",
      ].join(" ")}
      title="Click to add new token"
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      }}
      aria-label="Add new token"
    />
  );
}
