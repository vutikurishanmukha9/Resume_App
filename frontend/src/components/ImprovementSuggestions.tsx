import { ArrowUp, ArrowRight, ArrowDown } from "lucide-react";

interface ImprovementSuggestionsProps {
  suggestions: { title: string; description: string; impact: "high" | "medium" | "low" }[];
}

const impactConfig = {
  high: { icon: ArrowUp, label: "High Impact", className: "text-destructive bg-destructive/10 border-destructive/20" },
  medium: { icon: ArrowRight, label: "Medium", className: "text-warning bg-warning/10 border-warning/20" },
  low: { icon: ArrowDown, label: "Low", className: "text-muted-foreground bg-muted border-border" },
};

export const ImprovementSuggestions = ({ suggestions }: ImprovementSuggestionsProps) => {
  const safeSuggestions = suggestions || [];

  if (safeSuggestions.length === 0) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Improvement Suggestions</h3>
        <p className="text-sm text-muted-foreground">No specific improvements found. Your resume looks solid!</p>
      </div>
    );
  }

  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-semibold text-foreground mb-6">Improvement Suggestions</h3>

      <div className="grid md:grid-cols-2 gap-4">
        {safeSuggestions.map((s, i) => {
          const config = impactConfig[s.impact] || impactConfig.low;
          const Icon = config.icon;
          return (
            <div key={i} className="p-4 rounded-xl bg-secondary/50 border border-border hover:border-primary/20 transition-colors">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2.5">
                  <span className="w-6 h-6 rounded-md bg-primary/10 text-primary text-xs font-bold flex items-center justify-center shrink-0">
                    {i + 1}
                  </span>
                  <h4 className="text-sm font-semibold text-foreground">{s.title}</h4>
                </div>
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-medium border shrink-0 ${config.className}`}>
                  <Icon className="w-2.5 h-2.5" />
                  {config.label}
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed ml-8">{s.description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
