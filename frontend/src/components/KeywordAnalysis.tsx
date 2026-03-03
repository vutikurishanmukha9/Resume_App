import { AlertTriangle, AlertCircle, MinusCircle } from "lucide-react";

interface KeywordAnalysisProps {
  matched: string[];
  missing: { critical: string[]; important: string[]; optional: string[] };
}

export const KeywordAnalysis = ({ matched, missing }: KeywordAnalysisProps) => {
  const safeMatched = matched || [];
  const safeMissing = {
    critical: missing?.critical || [],
    important: missing?.important || [],
    optional: missing?.optional || [],
  };

  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-semibold text-foreground mb-6">Keyword Analysis</h3>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Matched */}
        <div>
          <p className="text-sm font-medium text-success mb-3 flex items-center gap-2">
            ✓ Matched Keywords
            <span className="text-xs font-normal text-muted-foreground">({safeMatched.length})</span>
          </p>
          <div className="flex flex-wrap gap-2">
            {safeMatched.map((kw) => (
              <span key={kw} className="keyword-tag-matched px-3 py-1 rounded-md text-xs font-medium">
                {kw}
              </span>
            ))}
          </div>
        </div>

        {/* Missing */}
        <div>
          <p className="text-sm font-medium text-destructive mb-3">✗ Missing Keywords</p>

          {safeMissing.critical.length > 0 && (
            <div className="mb-4">
              <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5">
                <AlertTriangle className="w-3 h-3 text-destructive" />
                Critical
              </p>
              <div className="flex flex-wrap gap-2">
                {safeMissing.critical.map((kw) => (
                  <span key={kw} className="keyword-tag-critical px-3 py-1 rounded-md text-xs font-medium">
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          )}

          {safeMissing.important.length > 0 && (
            <div className="mb-4">
              <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5">
                <AlertCircle className="w-3 h-3 text-warning" />
                Important
              </p>
              <div className="flex flex-wrap gap-2">
                {safeMissing.important.map((kw) => (
                  <span key={kw} className="keyword-tag-important px-3 py-1 rounded-md text-xs font-medium">
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          )}

          {safeMissing.optional.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1.5">
                <MinusCircle className="w-3 h-3 text-muted-foreground" />
                Optional
              </p>
              <div className="flex flex-wrap gap-2">
                {safeMissing.optional.map((kw) => (
                  <span key={kw} className="keyword-tag-optional px-3 py-1 rounded-md text-xs font-medium">
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
