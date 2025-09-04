import React, { useMemo, useState } from "react";
import EditableWordToken, { type WordToken } from "./editable-word-token";

export interface TokenRowProps {
  tokens: WordToken[];
  /** Called whenever any token changes. */
  onChange?: (next: WordToken[]) => void;
  className?: string;
}

/**
 * TokenRow renders a sequence of SimpleWordToken components with spaces between.
 * - Keeps array order stable.
 * - When a token's text changes, we also set `to_synth=true` if it differs from original.
 */
export default function TokenRow({ tokens, onChange, className }: TokenRowProps) {
  
  const handleTokenChange = (index: number) => (nextToken: WordToken) => {
    const adjusted: WordToken = {
      ...nextToken,
      // to_synth: nextToken.to_synth || nextToken.isEdited,
    };
    const next = tokens.map((t, i) => (i === index ? adjusted : t));
    onChange?.(next);
  };

  return (
    <div className={["token-row", className || ""].join(" ")}
         role="group"
         aria-label="Editable token row">
      {tokens.map((t, i) => (
        <React.Fragment key={`${t.start}-${t.end}-${i}`}>
          <EditableWordToken token={t} onChange={handleTokenChange(i)} />
          {i < tokens.length - 1 && (
            <span className="token-space" aria-hidden>
              {" "}
            </span>
          )}
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
