import React, { useMemo, useState, useRef, useEffect } from "react";
import EditableWordToken, { type WordToken, type EditableWordTokenRef } from "./editable-word-token";
import TokenSeparator from "./token-separator";

export interface TokenRowProps {
  tokens: WordToken[];
  /** Called whenever any token changes. */
  onChange?: (next: WordToken[]) => void;
  /** Called when a token is selected/edited. Receives the token index or null if no token is selected. */
  onTokenSelection?: (index: number | null) => void;
  className?: string;
}

/**
 * TokenRow renders a sequence of SimpleWordToken components with spaces between.
 * - Keeps array order stable.
 * - When a token's text changes, we also set `to_synth=true` if it differs from original.
 */
export default function TokenEditor({ tokens, onChange, onTokenSelection, className }: TokenRowProps) {
  const [newlyCreatedIndex, setNewlyCreatedIndex] = useState<number | null>(null);
  const tokenRefs = useRef<(EditableWordTokenRef | null)[]>([]);
  
  const handleTokenChange = (index: number) => (nextToken: WordToken) => {
    const adjusted: WordToken = {
      ...nextToken,
      // to_synth: nextToken.to_synth || nextToken.isEdited,
    };
    const next = tokens.map((t, i) => (i === index ? adjusted : t));
    onChange?.(next);
  };

  const handleTokenBlur = (index: number) => (token: WordToken) => {
    // Check if the token is empty and remove it from the array
    if (token.text.trim() === "") {
      const next = tokens.filter((_, i) => i !== index);
      onChange?.(next);
    }
    // Notify that no token is selected when blur occurs
    onTokenSelection?.(null);
  };

  const handleTokenFocus = (index: number) => () => {
    // Notify that this token is now selected
    onTokenSelection?.(index);
  };

  const createEmptyToken = (index: number): WordToken => {
    const prevToken = tokens[index - 1];
    const nextToken = tokens[index];
    
    // Calculate start and end times
    const start = prevToken ? prevToken.end : (nextToken ? nextToken.start : 0);
    const end = nextToken ? nextToken.start : (prevToken ? prevToken.end + 0.1 : 0.1);
    
    return {
      start,
      end,
      text: "",
      original_text: "",
      isEdited: true,
      predicted: false,
      to_synth: true,
      is_speech: true,
    };
  };

  const handleInsertToken = (index: number) => {
    const newToken = createEmptyToken(index);
    const next = [...tokens];
    next.splice(index, 0, newToken);
    setNewlyCreatedIndex(index);
    onChange?.(next);
  };

  // Focus newly created tokens
  useEffect(() => {
    if (newlyCreatedIndex !== null) {
      // Ensure refs array is the right size
      tokenRefs.current = tokenRefs.current.slice(0, tokens.length);
      
      // Focus the newly created token after a short delay to ensure DOM is updated
      const timeoutId = setTimeout(() => {
        const tokenRef = tokenRefs.current[newlyCreatedIndex];
        if (tokenRef) {
          tokenRef.focus();
        }
        setNewlyCreatedIndex(null);
      }, 50);
      
      return () => clearTimeout(timeoutId);
    }
  }, [newlyCreatedIndex, tokens.length]);

  return (
    <div className={["token-row", className || ""].join(" ")}
         role="group"
         aria-label="Editable token row">
      {tokens.map((t, i) => (
        <React.Fragment key={`${t.start}-${t.end}-${i}`}>
          <EditableWordToken 
            ref={(el) => (tokenRefs.current[i] = el)}
            token={t} 
            onChange={handleTokenChange(i)} 
            onBlur={handleTokenBlur(i)}
            onFocus={handleTokenFocus(i)}
          />
          <TokenSeparator onInsertToken={() => handleInsertToken(i + 1)} />
        </React.Fragment>
      ))}
      <style>{`
        .token-row {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.125rem; /* minimal inter-token gap; real space is rendered too */
          padding: 0.25rem;
          border: 1px solid #e5e7eb; /* neutral-200 */
          border-radius: 0.5rem;
          min-height: 2.5rem;
          cursor: text;
        }
        .token-space { white-space: pre; }
      `}</style>
    </div>
  );
}

// /**
//  * --- Demo component --------------------------------------------------------
//  * A quick example showing how to use TokenRow in a page.
//  */
// export function TokenRowDemo() {
//   const [tokens, setTokens] = useState<WordToken[]>([
//     { start: 0, end: 0.2, text: "The", original_text: "The", to_synth: false, is_speech: true },
//     { start: 0.2, end: 0.4, text: "tree", original_text: "tree", to_synth: false, is_speech: true },
//     { start: 0.4, end: 0.6, text: "is", original_text: "is", to_synth: false, is_speech: true },
//     { start: 0.6, end: 0.9, text: "[blank]", original_text: "[blank]", to_synth: true, is_speech: true },
//     { start: 0.9, end: 1.2, text: "tall.", original_text: "tall.", to_synth: false, is_speech: true },
//   ]);

//   const changedCount = useMemo(
//     () => tokens.filter(t => t.text !== (t.original_text ?? "")).length,
//     [tokens]
//   );

//   return (
//     <div>
//       <TokenRow tokens={tokens} onChange={setTokens} />
//       <div className="mt-2 text-sm text-neutral-600">Changed tokens: {changedCount}</div>
//       <pre className="mt-3 p-2 bg-neutral-50 border rounded text-xs overflow-auto">
//         {JSON.stringify(tokens, null, 2)}
//       </pre>
//     </div>
//   );
// }
