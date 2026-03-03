import { motion } from "framer-motion";
import { Download, Share2, RotateCcw, Briefcase, GraduationCap, Clock, Eye, Target, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScoreGauge } from "@/components/ScoreGauge";
import { KeywordAnalysis } from "@/components/KeywordAnalysis";
import { SkillsBreakdown } from "@/components/SkillsBreakdown";
import { ImprovementSuggestions } from "@/components/ImprovementSuggestions";
import { StrengthRadar } from "@/components/StrengthRadar";
import type { AnalysisResult } from "@/lib/types";

interface ResultsDashboardProps {
  result: AnalysisResult;
  onReanalyze: () => void;
}

const AnimatedStat = ({ value, label, suffix = "%" }: { value: number; label: string; suffix?: string }) => {
  return (
    <div className="text-center">
      <div className="text-2xl font-bold text-foreground">
        {value}{suffix}
      </div>
      <div className="text-xs text-muted-foreground mt-1">{label}</div>
    </div>
  );
};

const StatCard = ({ icon: Icon, label, value, className = "" }: { icon: any; label: string; value: string; className?: string }) => (
  <div className={`glass-card p-4 flex items-center gap-3 ${className}`}>
    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
      <Icon className="w-5 h-5 text-primary" />
    </div>
    <div className="min-w-0 flex-1">
      <p className="text-xs text-muted-foreground truncate">{label}</p>
      <p className="text-sm font-semibold text-foreground break-words truncate">{value}</p>
    </div>
  </div>
);

export const ResultsDashboard = ({ result, onReanalyze }: ResultsDashboardProps) => {
  const isHighScore = result.atsScore >= 80;

  return (
    <section className="pt-24 pb-20 px-6 hero-gradient min-h-screen">
      <div className="container mx-auto max-w-6xl">
        {/* Header actions */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-wrap items-center justify-between gap-4 mb-10"
        >
          <h2 className="text-2xl font-bold text-foreground">Analysis Results</h2>
          <div className="flex gap-2">
            <Button variant="glow" size="sm" onClick={onReanalyze}>
              <RotateCcw className="w-4 h-4 mr-1.5" />
              Re-analyze
            </Button>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-1.5" />
              PDF Report
            </Button>
            <Button variant="outline" size="sm">
              <Share2 className="w-4 h-4 mr-1.5" />
              Share
            </Button>
          </div>
        </motion.div>

        {/* Score + Stats row */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* ATS Score */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className={`glass-card p-8 flex flex-col items-center justify-center ${isHighScore ? "glow-success" : result.atsScore < 50 ? "glow-destructive" : ""
              }`}
          >
            <p className="text-xs text-muted-foreground font-medium mb-4 uppercase tracking-wider">ATS Score</p>
            <ScoreGauge score={result.atsScore} />
          </motion.div>

          {/* Stats grid */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-card p-6 flex flex-col justify-between gap-5"
          >
            <div className="grid grid-cols-3 gap-4">
              <AnimatedStat value={result.skillMatch} label="Skill Match" />
              <AnimatedStat value={result.recruiterReadability} label="Readability" />
              <AnimatedStat value={result.industryFit} label="Industry Fit" />
            </div>
            <div className="flex flex-col gap-3">
              <StatCard icon={Briefcase} label="Predicted Title" value={result.predictedTitle} />
              <StatCard icon={Clock} label="Experience" value={result.experience} />
              <StatCard icon={GraduationCap} label="Education" value={result.education} />
            </div>
          </motion.div>

          {/* Radar chart */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card p-6"
          >
            <p className="text-xs text-muted-foreground font-medium mb-3 uppercase tracking-wider">Resume Strength</p>
            <StrengthRadar data={result.strengthAreas} />
          </motion.div>
        </div>

        {/* Keyword Analysis */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-8"
        >
          <KeywordAnalysis matched={result.matchedKeywords} missing={result.missingKeywords} />
        </motion.div>

        {/* Skills Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mb-8"
        >
          <SkillsBreakdown skills={result.skills} />
        </motion.div>

        {/* Improvement Suggestions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <ImprovementSuggestions suggestions={result.suggestions} />
        </motion.div>
      </div>
    </section>
  );
};
