import { useState } from "react";
import { CheckCircle2, XCircle } from "lucide-react";

interface SkillsBreakdownProps {
  skills: Record<string, { matched: string[]; missing: string[] }>;
}

export const SkillsBreakdown = ({ skills }: SkillsBreakdownProps) => {
  const categories = Object.keys(skills || {});
  const [activeTab, setActiveTab] = useState(categories[0] || "");

  if (categories.length === 0) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Skills Breakdown</h3>
        <p className="text-sm text-muted-foreground">No skills data available. Try including a detailed job description for skills analysis.</p>
      </div>
    );
  }

  const current = skills[activeTab] || { matched: [], missing: [] };
  const total = (current.matched?.length || 0) + (current.missing?.length || 0);
  const pct = total > 0 ? Math.round(((current.matched?.length || 0) / total) * 100) : 0;

  return (
    <div className="glass-card p-6">
      <h3 className="text-lg font-semibold text-foreground mb-6">Skills Breakdown</h3>

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 mb-6">
        {categories.map((cat) => {
          const s = skills[cat] || { matched: [], missing: [] };
          const t = (s.matched?.length || 0) + (s.missing?.length || 0);
          const p = t > 0 ? Math.round(((s.matched?.length || 0) / t) * 100) : 0;
          return (
            <button
              key={cat}
              onClick={() => setActiveTab(cat)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${activeTab === cat
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                  : "bg-secondary text-muted-foreground hover:text-foreground"
                }`}
            >
              {cat}
              <span className="ml-2 text-xs opacity-70">{p}%</span>
            </button>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="mb-6">
        <div className="flex justify-between text-xs text-muted-foreground mb-2">
          <span>{current.matched?.length || 0} matched</span>
          <span>{current.missing?.length || 0} missing</span>
        </div>
        <div className="h-2 bg-secondary rounded-full overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-all duration-700 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Skills */}
      <div className="grid sm:grid-cols-2 gap-3">
        {(current.matched || []).map((skill) => (
          <div key={skill} className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-success/5 border border-success/15">
            <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
            <span className="text-sm text-foreground">{skill}</span>
          </div>
        ))}
        {(current.missing || []).map((skill) => (
          <div key={skill} className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-destructive/5 border border-destructive/15">
            <XCircle className="w-4 h-4 text-destructive shrink-0" />
            <span className="text-sm text-foreground">{skill}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
